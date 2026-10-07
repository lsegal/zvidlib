//! SIMD-accelerated pixel kernels shared by AV1 intra prediction and
//! reconstruction.
//!
//! The AV1 intra decoder, inter decoder, and [`crate::Av1IntraFrame`] all
//! reconstruct blocks the same way: derive a DC/vertical/horizontal/paeth
//! prediction for a row of samples, add the inverse-transformed residual, and
//! clamp back to 8-bit. Those inner loops are the hottest per-sample work in
//! the intra path, so they live here once behind a small kernel API instead of
//! being open-coded per decoder.
//!
//! Every kernel has a portable scalar implementation plus optional vector
//! implementations selected at run time:
//!
//! - x86/x86_64: AVX2, then SSE4.1, chosen with `is_x86_feature_detected!`.
//! - aarch64: NEON, which is part of the aarch64 baseline.
//! - Everything else (including `wasm32`, used by the browser decode path):
//!   the scalar fallback.
//!
//! The vector paths are bit-exact with the scalar ones; the test suite pins
//! that for every block size and prediction mode the decoders use.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::sync::OnceLock;

use crate::simd::SimdIsa;

/// The kernel implementation selected for this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1IntraSimd {
    /// The portable fallback, used when no vector implementation applies.
    Scalar,
    /// x86/x86_64 SSE4.1.
    Sse41,
    /// x86/x86_64 AVX2.
    Avx2,
    /// aarch64 NEON.
    Neon,
}

/// Reports which intra-prediction kernel implementation this process uses.
///
/// CPU feature detection runs at most once, but a [`crate::simd::set_override`]
/// override is consulted *ahead of* that cache on every call, so pinning an
/// instruction set (notably [`SimdIsa::Scalar`]) takes effect immediately even
/// after detection has already resolved.
pub fn av1_intra_simd() -> Av1IntraSimd {
    if let Some(isa) = crate::simd::override_isa() {
        return match isa {
            SimdIsa::Scalar => Av1IntraSimd::Scalar,
            SimdIsa::Sse41 => Av1IntraSimd::Sse41,
            SimdIsa::Avx2 => Av1IntraSimd::Avx2,
            SimdIsa::Neon => Av1IntraSimd::Neon,
        };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        static DETECTED: OnceLock<Av1IntraSimd> = OnceLock::new();
        *DETECTED.get_or_init(|| {
            if is_x86_feature_detected!("avx2") {
                Av1IntraSimd::Avx2
            } else if is_x86_feature_detected!("sse4.1") {
                Av1IntraSimd::Sse41
            } else {
                Av1IntraSimd::Scalar
            }
        })
    }
    #[cfg(target_arch = "aarch64")]
    {
        Av1IntraSimd::Neon
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Av1IntraSimd::Scalar
    }
}

/// Sums a contiguous run of neighbor samples, as the DC predictors do over the
/// row above and the column to the left of a block.
#[must_use]
#[inline]
pub fn sum_samples(samples: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match av1_intra_simd() {
        Av1IntraSimd::Avx2 => return unsafe { sum_samples_avx2(samples) },
        Av1IntraSimd::Sse41 => return unsafe { sum_samples_sse41(samples) },
        _ => {}
    }
    #[cfg(target_arch = "aarch64")]
    if av1_intra_simd() == Av1IntraSimd::Neon {
        return unsafe { sum_samples_neon(samples) };
    }
    sum_samples_scalar(samples)
}

/// Adds one row of residual samples to `row` in place and clamps the result
/// back to 8-bit, which is the final step of every reconstructed block.
///
/// # Panics
///
/// Panics when `residuals` and `row` have different lengths.
#[inline]
pub fn add_residual_row(residuals: &[i16], row: &mut [u8]) {
    assert_eq!(
        residuals.len(),
        row.len(),
        "AV1 residual row length must match the reconstruction row length"
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match av1_intra_simd() {
        Av1IntraSimd::Avx2 => return unsafe { add_residual_row_avx2(residuals, row) },
        Av1IntraSimd::Sse41 => return unsafe { add_residual_row_sse41(residuals, row) },
        _ => {}
    }
    #[cfg(target_arch = "aarch64")]
    if av1_intra_simd() == Av1IntraSimd::Neon {
        return unsafe { add_residual_row_neon(residuals, row) };
    }
    add_residual_row_scalar(residuals, row);
}

/// Writes one row of paeth predictions (spec §7.11.2.2) into `out`, given the
/// block's above-row samples, the single left-column sample for this row, and
/// the shared above-left sample.
///
/// # Panics
///
/// Panics when `top` and `out` have different lengths.
#[inline]
pub fn paeth_row(top_left: u8, top: &[u8], left: u8, out: &mut [u8]) {
    assert_eq!(
        top.len(),
        out.len(),
        "AV1 paeth prediction row length must match the above-row length"
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match av1_intra_simd() {
        Av1IntraSimd::Avx2 => return unsafe { paeth_row_avx2(top_left, top, left, out) },
        Av1IntraSimd::Sse41 => return unsafe { paeth_row_sse41(top_left, top, left, out) },
        _ => {}
    }
    #[cfg(target_arch = "aarch64")]
    if av1_intra_simd() == Av1IntraSimd::Neon {
        return unsafe { paeth_row_neon(top_left, top, left, out) };
    }
    paeth_row_scalar(top_left, top, left, out);
}

/// Writes one row of AV1 smooth-family intra predictions (spec §7.11.2.6).
#[inline]
pub fn smooth_row(mode: SmoothMode, top: &[u8], left: &[u8], row: usize, out: &mut [u8]) {
    assert_eq!(top.len(), out.len());
    assert!(row < left.len());
    assert!(smooth_weights(top.len()).is_some() && smooth_weights(left.len()).is_some());
    // No per-instruction-set arm: the §7.11.2.6 weighted blend has no vector
    // kernel yet, and routing it through one anyway is not free. A
    // `#[target_feature]` function can never be inlined into a caller that
    // does not carry the feature, so an arm that only forwards to
    // `smooth_row_scalar` still costs a call that cannot be merged into this
    // row loop - which on x86_64 measured 0.47-0.48x of calling the reference
    // directly, on both SSE4.1 and AVX2, for identical arithmetic (issue
    // #336). It read at parity on `neon` only because NEON is in the aarch64
    // baseline, so the same wrapper adds no features and inlines away. A real
    // kernel gets its arms back here; a placeholder does not earn them.
    smooth_row_scalar(mode, top, left, row, out);
}

/// Writes one row of AV1 directional intra predictions (spec §7.11.2.4).
#[inline]
pub fn directional_row(
    angle: i16,
    top: &[u8],
    left: &[u8],
    row: usize,
    filter_edges: bool,
    out: &mut [u8],
) {
    assert_eq!(top.len(), out.len());
    assert!(row < left.len());
    directional_row_scalar(angle, top, left, row, filter_edges, out);
}

/// The scalar paeth predictor for a single sample (spec §7.11.2.2).
#[must_use]
pub fn paeth(top_left: u8, top: u8, left: u8) -> u8 {
    let base = i16::from(top) + i16::from(left) - i16::from(top_left);
    let distance = |candidate: u8| (base - i16::from(candidate)).unsigned_abs();
    let top_distance = distance(top);
    let left_distance = distance(left);
    let corner_distance = distance(top_left);
    if top_distance <= left_distance && top_distance <= corner_distance {
        top
    } else if left_distance <= corner_distance {
        left
    } else {
        top_left
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmoothMode {
    Smooth,
    SmoothVertical,
    SmoothHorizontal,
}

// ---------------------------------------------------------------------
// Scalar reference implementations. The vector kernels below must stay
// bit-exact with these.

fn sum_samples_scalar(samples: &[u8]) -> u32 {
    samples.iter().map(|&sample| u32::from(sample)).sum()
}

fn add_residual_row_scalar(residuals: &[i16], row: &mut [u8]) {
    // The addition saturates rather than wraps so the scalar kernel stays
    // bit-exact with the vector paths (which use saturating 16-bit adds) even
    // for residuals near `i16::MIN`/`i16::MAX`, which no inverse transform in
    // this crate produces but which callers can hand to the public kernel.
    for (sample, &residual) in row.iter_mut().zip(residuals) {
        *sample = i16::from(*sample).saturating_add(residual).clamp(0, 255) as u8;
    }
}

fn paeth_row_scalar(top_left: u8, top: &[u8], left: u8, out: &mut [u8]) {
    for (prediction, &above) in out.iter_mut().zip(top) {
        *prediction = paeth(top_left, above, left);
    }
}

fn smooth_row_scalar(mode: SmoothMode, top: &[u8], left: &[u8], row: usize, out: &mut [u8]) {
    let weights_x = smooth_weights(top.len()).expect("validated smooth width");
    let weights_y = smooth_weights(left.len()).expect("validated smooth height");
    let bottom_left = u32::from(left[left.len() - 1]);
    let top_right = u32::from(top[top.len() - 1]);
    for (column, prediction) in out.iter_mut().enumerate() {
        let above = u32::from(top[column]);
        let left_sample = u32::from(left[row]);
        let value = match mode {
            SmoothMode::Smooth => {
                u32::from(weights_y[row]) * above
                    + u32::from(256u16 - u16::from(weights_y[row])) * bottom_left
                    + u32::from(weights_x[column]) * left_sample
                    + u32::from(256u16 - u16::from(weights_x[column])) * top_right
                    + 256
            }
            SmoothMode::SmoothVertical => {
                u32::from(weights_y[row]) * above
                    + u32::from(256u16 - u16::from(weights_y[row])) * bottom_left
                    + 128
            }
            SmoothMode::SmoothHorizontal => {
                u32::from(weights_x[column]) * left_sample
                    + u32::from(256u16 - u16::from(weights_x[column])) * top_right
                    + 128
            }
        };
        *prediction = (value
            >> match mode {
                SmoothMode::Smooth => 9,
                SmoothMode::SmoothVertical | SmoothMode::SmoothHorizontal => 8,
            }) as u8;
    }
}

fn directional_row_scalar(
    angle: i16,
    top: &[u8],
    left: &[u8],
    row: usize,
    filter_edges: bool,
    out: &mut [u8],
) {
    let width = top.len();
    let height = left.len();
    let angle = angle.clamp(0, 270);
    if angle == 90 {
        out.copy_from_slice(top);
        return;
    }
    if angle == 180 {
        out.fill(left[row]);
        return;
    }
    let mut above = Edge::from(top, left[0]);
    let mut left_edge = Edge::from(left, top[0]);
    let mut upsample_above = false;
    let mut upsample_left = false;
    if filter_edges {
        let filter_type = 0;
        if angle > 90 && angle < 180 && width + height >= 24 {
            let corner = ((u16::from(left[0]) * 5
                + u16::from(above.get(-1)) * 6
                + u16::from(top[0]) * 5
                + 8)
                >> 4) as u8;
            above.set(-1, corner);
            left_edge.set(-1, corner);
        }
        if angle < 180 {
            let strength = intra_edge_filter_strength(width, height, 0, i32::from(angle) - 90);
            above.filter(width + usize::from(angle < 90) * height + 1, strength);
        }
        if angle > 90 {
            let strength = intra_edge_filter_strength(width, height, 0, i32::from(angle) - 180);
            left_edge.filter(height + usize::from(angle > 180) * width + 1, strength);
        }
        upsample_above = intra_edge_upsample(width, height, filter_type, i32::from(angle) - 90);
        if upsample_above {
            above.upsample(width + usize::from(angle < 90) * height);
        }
        upsample_left = intra_edge_upsample(width, height, filter_type, i32::from(angle) - 180);
        if upsample_left {
            left_edge.upsample(height + usize::from(angle > 180) * width);
        }
    }
    if angle < 90 {
        let dx = dr_intra_derivative(angle);
        let idx = (row as i32 + 1) * dx;
        let max_base_x = ((width + height - 1) as i32) << i32::from(upsample_above);
        for (column, prediction) in out.iter_mut().enumerate() {
            let base = (idx >> (6 - u32::from(upsample_above)))
                + ((column as i32) << i32::from(upsample_above));
            let shift = (((idx << i32::from(upsample_above)) >> 1) & 0x1f) as u16;
            *prediction = if base < max_base_x {
                round_weighted(above.get(base), above.get(base + 1), shift)
            } else {
                above.get(max_base_x)
            };
        }
    } else if angle < 180 {
        let dx = dr_intra_derivative(180 - angle);
        let dy = dr_intra_derivative(angle - 90);
        for (column, prediction) in out.iter_mut().enumerate() {
            let idx = ((column as i32) << 6) - (row as i32 + 1) * dx;
            let base = idx >> (6 - u32::from(upsample_above));
            if base >= -(1 << i32::from(upsample_above)) {
                let shift = (((idx << i32::from(upsample_above)) >> 1) & 0x1f) as u16;
                *prediction = round_weighted(above.get(base), above.get(base + 1), shift);
            } else {
                let idx = ((row as i32) << 6) - (column as i32 + 1) * dy;
                let base = idx >> (6 - u32::from(upsample_left));
                let shift = (((idx << i32::from(upsample_left)) >> 1) & 0x1f) as u16;
                *prediction = round_weighted(left_edge.get(base), left_edge.get(base + 1), shift);
            }
        }
    } else {
        let dy = dr_intra_derivative(270 - angle);
        for (column, prediction) in out.iter_mut().enumerate() {
            let idx = (column as i32 + 1) * dy;
            let base = (idx >> (6 - u32::from(upsample_left)))
                + ((row as i32) << i32::from(upsample_left));
            let shift = (((idx << i32::from(upsample_left)) >> 1) & 0x1f) as u16;
            *prediction = round_weighted(left_edge.get(base), left_edge.get(base + 1), shift);
        }
    }
}

fn round_weighted(a: u8, b: u8, shift: u16) -> u8 {
    ((u16::from(a) * (32 - shift) + u16::from(b) * shift + 16) >> 5) as u8
}

fn smooth_weights(size: usize) -> Option<&'static [u8]> {
    match size {
        4 => Some(&SM_WEIGHTS_4),
        8 => Some(&SM_WEIGHTS_8),
        16 => Some(&SM_WEIGHTS_16),
        32 => Some(&SM_WEIGHTS_32),
        64 | 128 => Some(&SM_WEIGHTS_64),
        _ => None,
    }
}

fn dr_intra_derivative(angle: i16) -> i32 {
    const DERIVATIVE: [i32; 90] = [
        0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0,
        0, 132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0,
        51, 0, 0, 45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0,
        0, 0, 0, 11, 0, 0, 7, 0, 0, 3, 0, 0,
    ];
    DERIVATIVE.get(angle as usize).copied().unwrap_or(0)
}

fn intra_edge_filter_strength(w: usize, h: usize, filter_type: usize, delta: i32) -> usize {
    let d = delta.unsigned_abs();
    let blk_wh = w + h;
    let mut strength = 0;
    if filter_type == 0 {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

fn intra_edge_upsample(w: usize, h: usize, filter_type: usize, delta: i32) -> bool {
    let d = delta.unsigned_abs();
    if d == 0 || d >= 40 {
        false
    } else if filter_type == 0 {
        w + h <= 16
    } else {
        w + h <= 8
    }
}

struct Edge {
    base: i32,
    samples: Vec<u8>,
}

impl Edge {
    fn from(edge: &[u8], fallback: u8) -> Self {
        let mut samples = Vec::with_capacity(edge.len() * 2 + 4);
        let last = edge.last().copied().unwrap_or(fallback);
        samples.extend([fallback, fallback]);
        samples.extend_from_slice(edge);
        samples.resize(edge.len() * 2 + 4, last);
        Self { base: -2, samples }
    }

    fn get(&self, index: i32) -> u8 {
        let slot = index - self.base;
        if slot < 0 {
            self.samples[0]
        } else {
            self.samples
                .get(slot as usize)
                .copied()
                .unwrap_or_else(|| *self.samples.last().expect("edge is non-empty"))
        }
    }

    fn set(&mut self, index: i32, value: u8) {
        let slot = index - self.base;
        if slot >= 0 {
            if let Some(sample) = self.samples.get_mut(slot as usize) {
                *sample = value;
            }
        }
    }

    fn filter(&mut self, size: usize, strength: usize) {
        if strength == 0 {
            return;
        }
        const KERNELS: [[u16; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
        let edge: Vec<u8> = (0..size).map(|i| self.get(i as i32 - 1)).collect();
        let kernel = &KERNELS[strength - 1];
        for i in 1..size {
            let mut sum = 0u16;
            for (j, weight) in kernel.iter().enumerate() {
                let k = (i as isize - 2 + j as isize).clamp(0, size as isize - 1) as usize;
                sum += *weight * u16::from(edge[k]);
            }
            self.set(i as i32 - 1, ((sum + 8) >> 4) as u8);
        }
    }

    fn upsample(&mut self, num_px: usize) {
        let dup: Vec<u8> = std::iter::once(self.get(-1))
            .chain((-1..num_px as i32).map(|i| self.get(i)))
            .chain(std::iter::once(self.get(num_px as i32 - 1)))
            .collect();
        self.base = -2;
        self.samples.resize(num_px * 2 + 1, 0);
        self.samples[0] = dup[0];
        for i in 0..num_px {
            let s = -i16::from(dup[i]) + 9 * i16::from(dup[i + 1]) + 9 * i16::from(dup[i + 2])
                - i16::from(dup[i + 3]);
            self.set(2 * i as i32 - 1, ((s + 8) >> 4).clamp(0, 255) as u8);
            self.set(2 * i as i32, dup[i + 2]);
        }
    }
}

const SM_WEIGHTS_4: [u8; 4] = [255, 149, 85, 64];
const SM_WEIGHTS_8: [u8; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
const SM_WEIGHTS_16: [u8; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
const SM_WEIGHTS_32: [u8; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
const SM_WEIGHTS_64: [u8; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133,
    127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38,
    35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

// ---------------------------------------------------------------------
// x86/x86_64.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sum_samples_avx2(samples: &[u8]) -> u32 {
    unsafe {
        let mut index = 0;
        let mut accumulator = _mm256_setzero_si256();
        let zero = _mm256_setzero_si256();
        while index + 32 <= samples.len() {
            let chunk = _mm256_loadu_si256(samples.as_ptr().add(index).cast());
            accumulator = _mm256_add_epi64(accumulator, _mm256_sad_epu8(chunk, zero));
            index += 32;
        }
        let mut lanes = [0i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), accumulator);
        let total: i64 = lanes.iter().sum();
        total as u32 + sum_samples_scalar(&samples[index..])
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn sum_samples_sse41(samples: &[u8]) -> u32 {
    unsafe {
        let mut index = 0;
        let mut accumulator = _mm_setzero_si128();
        let zero = _mm_setzero_si128();
        while index + 16 <= samples.len() {
            let chunk = _mm_loadu_si128(samples.as_ptr().add(index).cast());
            accumulator = _mm_add_epi64(accumulator, _mm_sad_epu8(chunk, zero));
            index += 16;
        }
        let mut lanes = [0i64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), accumulator);
        (lanes[0] + lanes[1]) as u32 + sum_samples_scalar(&samples[index..])
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_residual_row_avx2(residuals: &[i16], row: &mut [u8]) {
    unsafe {
        let mut index = 0;
        while index + 16 <= row.len() {
            let samples = _mm_loadu_si128(row.as_ptr().add(index).cast());
            let widened = _mm256_cvtepu8_epi16(samples);
            let residual = _mm256_loadu_si256(residuals.as_ptr().add(index).cast());
            // Saturating i16 arithmetic then unsigned-saturating packing is
            // exactly `(sample + residual).clamp(0, 255)`.
            let sum = _mm256_adds_epi16(widened, residual);
            let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(sum, sum), 0b0000_1000);
            _mm_storeu_si128(
                row.as_mut_ptr().add(index).cast(),
                _mm256_castsi256_si128(packed),
            );
            index += 16;
        }
        add_residual_row_scalar(&residuals[index..], &mut row[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn add_residual_row_sse41(residuals: &[i16], row: &mut [u8]) {
    unsafe {
        let mut index = 0;
        while index + 8 <= row.len() {
            let samples = _mm_loadl_epi64(row.as_ptr().add(index).cast());
            let widened = _mm_cvtepu8_epi16(samples);
            let residual = _mm_loadu_si128(residuals.as_ptr().add(index).cast());
            let sum = _mm_adds_epi16(widened, residual);
            let packed = _mm_packus_epi16(sum, sum);
            _mm_storel_epi64(row.as_mut_ptr().add(index).cast(), packed);
            index += 8;
        }
        add_residual_row_scalar(&residuals[index..], &mut row[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn paeth_row_avx2(top_left: u8, top: &[u8], left: u8, out: &mut [u8]) {
    unsafe {
        let corner = _mm256_set1_epi16(i16::from(top_left));
        let left_lanes = _mm256_set1_epi16(i16::from(left));
        let mut index = 0;
        while index + 16 <= top.len() {
            let above = _mm256_cvtepu8_epi16(_mm_loadu_si128(top.as_ptr().add(index).cast()));
            let base = _mm256_sub_epi16(_mm256_add_epi16(above, left_lanes), corner);
            let top_distance = _mm256_abs_epi16(_mm256_sub_epi16(base, above));
            let left_distance = _mm256_abs_epi16(_mm256_sub_epi16(base, left_lanes));
            let corner_distance = _mm256_abs_epi16(_mm256_sub_epi16(base, corner));
            // `a <= b` is the complement of `a > b`; blendv selects the second
            // operand wherever the mask's high bit is set.
            let prefer_top = _mm256_andnot_si256(
                _mm256_or_si256(
                    _mm256_cmpgt_epi16(top_distance, left_distance),
                    _mm256_cmpgt_epi16(top_distance, corner_distance),
                ),
                _mm256_set1_epi16(-1),
            );
            let prefer_left = _mm256_andnot_si256(
                _mm256_cmpgt_epi16(left_distance, corner_distance),
                _mm256_set1_epi16(-1),
            );
            let chosen = _mm256_blendv_epi8(
                _mm256_blendv_epi8(corner, left_lanes, prefer_left),
                above,
                prefer_top,
            );
            let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(chosen, chosen), 0b0000_1000);
            _mm_storeu_si128(
                out.as_mut_ptr().add(index).cast(),
                _mm256_castsi256_si128(packed),
            );
            index += 16;
        }
        paeth_row_scalar(top_left, &top[index..], left, &mut out[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn paeth_row_sse41(top_left: u8, top: &[u8], left: u8, out: &mut [u8]) {
    unsafe {
        let corner = _mm_set1_epi16(i16::from(top_left));
        let left_lanes = _mm_set1_epi16(i16::from(left));
        let ones = _mm_set1_epi16(-1);
        let mut index = 0;
        while index + 8 <= top.len() {
            let above = _mm_cvtepu8_epi16(_mm_loadl_epi64(top.as_ptr().add(index).cast()));
            let base = _mm_sub_epi16(_mm_add_epi16(above, left_lanes), corner);
            let top_distance = _mm_abs_epi16(_mm_sub_epi16(base, above));
            let left_distance = _mm_abs_epi16(_mm_sub_epi16(base, left_lanes));
            let corner_distance = _mm_abs_epi16(_mm_sub_epi16(base, corner));
            let prefer_top = _mm_andnot_si128(
                _mm_or_si128(
                    _mm_cmpgt_epi16(top_distance, left_distance),
                    _mm_cmpgt_epi16(top_distance, corner_distance),
                ),
                ones,
            );
            let prefer_left =
                _mm_andnot_si128(_mm_cmpgt_epi16(left_distance, corner_distance), ones);
            let chosen = _mm_blendv_epi8(
                _mm_blendv_epi8(corner, left_lanes, prefer_left),
                above,
                prefer_top,
            );
            _mm_storel_epi64(
                out.as_mut_ptr().add(index).cast(),
                _mm_packus_epi16(chosen, chosen),
            );
            index += 8;
        }
        paeth_row_scalar(top_left, &top[index..], left, &mut out[index..]);
    }
}

// ---------------------------------------------------------------------
// aarch64 NEON.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_samples_neon(samples: &[u8]) -> u32 {
    unsafe {
        let mut index = 0;
        let mut total = 0u32;
        while index + 16 <= samples.len() {
            total += u32::from(vaddlvq_u8(vld1q_u8(samples.as_ptr().add(index))));
            index += 16;
        }
        total + sum_samples_scalar(&samples[index..])
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_residual_row_neon(residuals: &[i16], row: &mut [u8]) {
    unsafe {
        let mut index = 0;
        // 16 samples per iteration matches what the autovectorizer produces
        // for the scalar loop, which matters for the 64- and 128-wide blocks.
        while index + 16 <= row.len() {
            let samples = vld1q_u8(row.as_ptr().add(index));
            let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(samples)));
            let high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(samples)));
            let sum_low = vqaddq_s16(low, vld1q_s16(residuals.as_ptr().add(index)));
            let sum_high = vqaddq_s16(high, vld1q_s16(residuals.as_ptr().add(index + 8)));
            vst1q_u8(
                row.as_mut_ptr().add(index),
                vcombine_u8(vqmovun_s16(sum_low), vqmovun_s16(sum_high)),
            );
            index += 16;
        }
        while index + 8 <= row.len() {
            let widened = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row.as_ptr().add(index))));
            let residual = vld1q_s16(residuals.as_ptr().add(index));
            // Saturating add plus unsigned-saturating narrowing is exactly
            // `(sample + residual).clamp(0, 255)`.
            let sum = vqaddq_s16(widened, residual);
            vst1_u8(row.as_mut_ptr().add(index), vqmovun_s16(sum));
            index += 8;
        }
        add_residual_row_scalar(&residuals[index..], &mut row[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn paeth_row_neon(top_left: u8, top: &[u8], left: u8, out: &mut [u8]) {
    unsafe {
        let corner = vdupq_n_s16(i16::from(top_left));
        let left_lanes = vdupq_n_s16(i16::from(left));
        let mut index = 0;
        while index + 8 <= top.len() {
            let above = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(top.as_ptr().add(index))));
            let base = vsubq_s16(vaddq_s16(above, left_lanes), corner);
            let top_distance = vabdq_s16(base, above);
            let left_distance = vabdq_s16(base, left_lanes);
            let corner_distance = vabdq_s16(base, corner);
            let prefer_top = vandq_u16(
                vcleq_s16(top_distance, left_distance),
                vcleq_s16(top_distance, corner_distance),
            );
            let prefer_left = vcleq_s16(left_distance, corner_distance);
            let chosen = vbslq_s16(
                prefer_top,
                above,
                vbslq_s16(prefer_left, left_lanes, corner),
            );
            vst1_u8(out.as_mut_ptr().add(index), vqmovun_s16(chosen));
            index += 8;
        }
        paeth_row_scalar(top_left, &top[index..], left, &mut out[index..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed
    }

    #[test]
    fn sum_samples_matches_the_scalar_kernel() {
        let mut seed = 7;
        let samples: Vec<u8> = (0..133)
            .map(|_| (pseudo_random(&mut seed) >> 13) as u8)
            .collect();
        for length in 0..samples.len() {
            assert_eq!(
                sum_samples(&samples[..length]),
                sum_samples_scalar(&samples[..length]),
                "length {length}"
            );
        }
    }

    #[test]
    fn add_residual_row_matches_the_scalar_kernel() {
        let mut seed = 11;
        let base: Vec<u8> = (0..71)
            .map(|_| (pseudo_random(&mut seed) >> 11) as u8)
            .collect();
        let residuals: Vec<i16> = (0..71)
            .map(|_| ((pseudo_random(&mut seed) >> 9) as i16).wrapping_sub(600))
            .collect();
        for length in 0..base.len() {
            let mut expected = base[..length].to_vec();
            add_residual_row_scalar(&residuals[..length], &mut expected);
            let mut actual = base[..length].to_vec();
            add_residual_row(&residuals[..length], &mut actual);
            assert_eq!(actual, expected, "length {length}");
        }
    }

    #[test]
    fn add_residual_row_saturates_at_both_ends() {
        let mut row = [0u8, 255, 128, 3, 250, 1, 200, 40, 17];
        let residuals = [-1i16, 1, 32_767, -32_768, 10, -10, 60, 300, -300];
        let mut expected = row;
        add_residual_row_scalar(&residuals, &mut expected);
        add_residual_row(&residuals, &mut row);
        assert_eq!(row, expected);
        assert_eq!(expected, [0, 255, 255, 0, 255, 0, 255, 255, 0]);
    }

    #[test]
    fn paeth_row_matches_the_scalar_kernel() {
        let mut seed = 29;
        let top: Vec<u8> = (0..53)
            .map(|_| (pseudo_random(&mut seed) >> 15) as u8)
            .collect();
        for &top_left in &[0u8, 1, 127, 128, 200, 255] {
            for &left in &[0u8, 3, 64, 128, 254, 255] {
                for length in 0..top.len() {
                    let mut expected = vec![0u8; length];
                    paeth_row_scalar(top_left, &top[..length], left, &mut expected);
                    let mut actual = vec![0u8; length];
                    paeth_row(top_left, &top[..length], left, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "top_left {top_left} left {left} length {length}"
                    );
                }
            }
        }
    }

    #[test]
    fn paeth_row_covers_every_sample_combination() {
        let top: Vec<u8> = (0..=255).collect();
        for top_left in 0..=255u8 {
            for left in [0u8, 17, 128, 255] {
                let mut expected = vec![0u8; top.len()];
                paeth_row_scalar(top_left, &top, left, &mut expected);
                let mut actual = vec![0u8; top.len()];
                paeth_row(top_left, &top, left, &mut actual);
                assert_eq!(actual, expected, "top_left {top_left} left {left}");
            }
        }
    }
}
