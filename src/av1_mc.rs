//! AV1 inter prediction primitives (motion compensation) with SIMD backends.
//!
//! This module implements the pixel-processing kernels that dominate AV1
//! inter-frame decode cost, split out from the bitstream-driven decoder in
//! [`crate::av1_inter_decoder`] so they can be vectorized and tested on
//! their own:
//!
//! * **Sub-pixel interpolation** (spec §7.11.3.4, *Block Inter Prediction
//!   Process*): the separable two-pass convolution using the AV1
//!   `Subpel_Filters` tap sets, at the 1/16-pel phase resolution the spec
//!   defines.
//! * **Compound prediction blending** (spec §7.11.3.1 and §7.11.3.14): the
//!   simple average, the distance-weighted average (§7.11.3.15), and the
//!   masked blend consumed by both wedge and difference-weighted compound
//!   modes, plus the difference-weighted mask generation of §7.11.3.12.
//!
//! # Backends
//!
//! Every kernel has a scalar reference implementation plus vectorized
//! implementations selected at run time:
//!
//! | Target | Backends |
//! | --- | --- |
//! | `x86_64` | AVX2, SSE4.1, scalar |
//! | `aarch64` | NEON, scalar |
//! | anything else (including `wasm32`) | scalar |
//!
//! Selection goes through [`detected_level`], which caches the result of
//! `is_x86_feature_detected!` (NEON is architecturally guaranteed on
//! `aarch64`, so it needs no probe). Every vectorized kernel is required to
//! be *bit-exact* with the scalar one; the tests at the bottom of this file
//! assert that across every block size, filter, sub-pel phase, and compound
//! mode this module supports, for every backend the host can execute.
//!
//! # Fixed-point conventions
//!
//! The spec's rounding variables (§7.11.3.2) for 8-bit samples are
//! `InterRound0 = 3`, `InterRound1 = 11` for single prediction and `7` for
//! compound prediction, with `InterPostRound = 2 * FILTER_BITS -
//! InterRound0 - InterRound1 = 4` applied when compound predictions are
//! blended. Single prediction therefore lands directly on final 8-bit
//! pixels, while compound prediction produces intermediate `i16` samples at
//! 16x pixel scale which the blend kernels round back down.

use std::sync::OnceLock;

/// Number of fractional bits in the interpolation filter taps (spec
/// `FILTER_BITS`).
pub const FILTER_BITS: u32 = 7;

/// Number of distinct sub-pel phases (spec `SUBPEL_SHIFTS`): AV1 filters at
/// 1/16-pel resolution.
pub const SUBPEL_SHIFTS: usize = 16;

/// Spec `InterRound0`: rounding applied after the horizontal pass.
const INTER_ROUND0: u32 = 3;
/// Spec `InterRound1` for single (non-compound) prediction.
const INTER_ROUND1_SINGLE: u32 = 11;
/// Spec `InterRound1` for compound prediction.
const INTER_ROUND1_COMPOUND: u32 = 7;
/// Spec `InterPostRound`: `2 * FILTER_BITS - InterRound0 - InterRound1`.
const INTER_POST_ROUND: u32 = 2 * FILTER_BITS - INTER_ROUND0 - INTER_ROUND1_COMPOUND;

/// Maximum blend weight for masked compound prediction (spec
/// `MASK_MASTER_SIZE` blending uses a 0..=64 alpha).
pub const MAX_MASK_ALPHA: u8 = 64;

/// Base mask value for difference-weighted compound masks (spec §7.11.3.12).
const DIFF_MASK_BASE: i32 = 38;
/// Divisor applied to the rounded prediction difference (spec §7.11.3.12).
const DIFF_MASK_FACTOR: i32 = 16;

/// Spec `MAX_FRAME_DISTANCE`: order-hint distances saturate here before the
/// distance-weighted compound weights are looked up.
const MAX_FRAME_DISTANCE: u32 = 31;

/// Half the filter support: taps span `[-3, +4]` around the integer sample.
const FILTER_LEFT: usize = 3;
/// Extra samples a filtered block needs beyond its own width/height.
const FILTER_MARGIN: usize = 7;

/// `Round2(x, n)` from the spec, for signed values.
#[inline(always)]
const fn round2(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        value
    } else {
        (value + (1 << (shift - 1))) >> shift
    }
}

/// `Clip1(x)` for 8-bit samples.
#[inline(always)]
const fn clip_pixel(value: i32) -> u8 {
    if value < 0 {
        0
    } else if value > 255 {
        255
    } else {
        value as u8
    }
}

/// The interpolation filter sets AV1 signals through `interpolation_filter`
/// (spec §6.8.9).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpFilter {
    /// `EIGHTTAP_REGULAR`.
    Regular,
    /// `EIGHTTAP_SMOOTH`.
    Smooth,
    /// `EIGHTTAP_SHARP`.
    Sharp,
    /// `BILINEAR`.
    Bilinear,
}

impl InterpFilter {
    /// All four filters, for exhaustive testing.
    pub const ALL: [InterpFilter; 4] = [
        InterpFilter::Regular,
        InterpFilter::Smooth,
        InterpFilter::Sharp,
        InterpFilter::Bilinear,
    ];

    /// Maps `interpolation_filter` as coded in the frame header.
    pub fn from_code(code: u32) -> Option<InterpFilter> {
        match code {
            0 => Some(InterpFilter::Regular),
            1 => Some(InterpFilter::Smooth),
            2 => Some(InterpFilter::Sharp),
            3 => Some(InterpFilter::Bilinear),
            _ => None,
        }
    }

    /// Index into [`SUBPEL_FILTERS`] for this filter at the given block
    /// dimension. Spec §7.11.3.4 substitutes the narrow 4-tap tap sets for
    /// blocks whose relevant dimension is at most four samples wide, which
    /// keeps small blocks from reading (and weighting) samples that are
    /// mostly outside the block.
    const fn tap_set(self, size: usize) -> usize {
        match self {
            InterpFilter::Regular => {
                if size <= 4 {
                    3
                } else {
                    0
                }
            }
            InterpFilter::Smooth => {
                if size <= 4 {
                    4
                } else {
                    1
                }
            }
            InterpFilter::Sharp => {
                if size <= 4 {
                    3
                } else {
                    2
                }
            }
            InterpFilter::Bilinear => 5,
        }
    }

    /// The eight taps to convolve with for `subpel` (0..[`SUBPEL_SHIFTS`])
    /// on a block of `size` samples along the filtered axis.
    ///
    /// # Panics
    ///
    /// Panics if `subpel` is not a valid sub-pel phase.
    pub fn taps(self, size: usize, subpel: usize) -> &'static [i16; 8] {
        assert!(subpel < SUBPEL_SHIFTS, "sub-pel phase out of range");
        &SUBPEL_FILTERS[self.tap_set(size)][subpel]
    }
}

/// The AV1 `Subpel_Filters` table (spec §7.11.3.4). Row order is
/// `EIGHTTAP_REGULAR`, `EIGHTTAP_SMOOTH`, `EIGHTTAP_SHARP`, the two narrow
/// 4-tap sets used for blocks of four samples or fewer, then `BILINEAR`.
/// Every phase sums to `1 << FILTER_BITS` so a flat input is reproduced
/// exactly.
#[rustfmt::skip]
pub static SUBPEL_FILTERS: [[[i16; 8]; SUBPEL_SHIFTS]; 6] = [
    // EIGHTTAP_REGULAR
    [
        [0, 0, 0, 128, 0, 0, 0, 0],       [0, 2, -6, 126, 8, -2, 0, 0],
        [0, 2, -10, 122, 18, -4, 0, 0],   [0, 2, -12, 116, 28, -8, 2, 0],
        [0, 2, -14, 110, 38, -10, 2, 0],  [0, 2, -14, 102, 48, -12, 2, 0],
        [0, 2, -16, 94, 58, -12, 2, 0],   [0, 2, -14, 84, 66, -12, 2, 0],
        [0, 2, -14, 76, 76, -14, 2, 0],   [0, 2, -12, 66, 84, -14, 2, 0],
        [0, 2, -12, 58, 94, -16, 2, 0],   [0, 2, -12, 48, 102, -14, 2, 0],
        [0, 2, -10, 38, 110, -14, 2, 0],  [0, 2, -8, 28, 116, -12, 2, 0],
        [0, 0, -4, 18, 122, -10, 2, 0],   [0, 0, -2, 8, 126, -6, 2, 0],
    ],
    // EIGHTTAP_SMOOTH
    [
        [0, 0, 0, 128, 0, 0, 0, 0],       [0, 2, 28, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],      [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],      [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],     [0, -2, 16, 54, 48, 12, 0, 0],
        [0, -2, 14, 52, 52, 14, -2, 0],   [0, 0, 12, 48, 54, 16, -2, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],     [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],      [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],      [0, 0, 2, 34, 62, 28, 2, 0],
    ],
    // EIGHTTAP_SHARP
    [
        [0, 0, 0, 128, 0, 0, 0, 0],           [-2, 2, -6, 126, 8, -2, 2, 0],
        [-2, 6, -12, 124, 16, -6, 4, -2],     [-2, 8, -18, 120, 26, -10, 6, -2],
        [-4, 10, -22, 116, 38, -14, 6, -2],   [-4, 10, -22, 108, 48, -18, 8, -2],
        [-4, 10, -24, 100, 60, -20, 8, -2],   [-4, 10, -24, 90, 70, -22, 10, -2],
        [-4, 12, -24, 80, 80, -24, 12, -4],   [-2, 10, -22, 70, 90, -24, 10, -4],
        [-2, 8, -20, 60, 100, -24, 10, -4],   [-2, 8, -18, 48, 108, -22, 10, -4],
        [-2, 6, -14, 38, 116, -22, 10, -4],   [-2, 6, -10, 26, 120, -18, 8, -2],
        [-2, 4, -6, 16, 124, -12, 6, -2],     [0, 2, -2, 8, 126, -6, 2, -2],
    ],
    // 4-tap regular (also used for sharp on narrow blocks)
    [
        [0, 0, 0, 128, 0, 0, 0, 0],       [0, 0, -4, 126, 8, -2, 0, 0],
        [0, 0, -8, 122, 18, -4, 0, 0],    [0, 0, -10, 116, 28, -6, 0, 0],
        [0, 0, -12, 110, 38, -8, 0, 0],   [0, 0, -12, 102, 48, -10, 0, 0],
        [0, 0, -14, 94, 58, -10, 0, 0],   [0, 0, -12, 84, 66, -10, 0, 0],
        [0, 0, -12, 76, 76, -12, 0, 0],   [0, 0, -10, 66, 84, -12, 0, 0],
        [0, 0, -10, 58, 94, -14, 0, 0],   [0, 0, -10, 48, 102, -12, 0, 0],
        [0, 0, -8, 38, 110, -12, 0, 0],   [0, 0, -6, 28, 116, -10, 0, 0],
        [0, 0, -4, 18, 122, -8, 0, 0],    [0, 0, -2, 8, 126, -4, 0, 0],
    ],
    // 4-tap smooth
    [
        [0, 0, 0, 128, 0, 0, 0, 0],       [0, 0, 30, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],      [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],      [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],     [0, 0, 14, 54, 48, 12, 0, 0],
        [0, 0, 12, 52, 52, 12, 0, 0],     [0, 0, 12, 48, 54, 14, 0, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],     [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],      [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],      [0, 0, 2, 34, 62, 30, 0, 0],
    ],
    // BILINEAR
    [
        [0, 0, 0, 128, 0, 0, 0, 0],       [0, 0, 0, 120, 8, 0, 0, 0],
        [0, 0, 0, 112, 16, 0, 0, 0],      [0, 0, 0, 104, 24, 0, 0, 0],
        [0, 0, 0, 96, 32, 0, 0, 0],       [0, 0, 0, 88, 40, 0, 0, 0],
        [0, 0, 0, 80, 48, 0, 0, 0],       [0, 0, 0, 72, 56, 0, 0, 0],
        [0, 0, 0, 64, 64, 0, 0, 0],       [0, 0, 0, 56, 72, 0, 0, 0],
        [0, 0, 0, 48, 80, 0, 0, 0],       [0, 0, 0, 40, 88, 0, 0, 0],
        [0, 0, 0, 32, 96, 0, 0, 0],       [0, 0, 0, 24, 104, 0, 0, 0],
        [0, 0, 0, 16, 112, 0, 0, 0],      [0, 0, 0, 8, 120, 0, 0, 0],
    ],
];

/// Which instruction set a kernel should run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdLevel {
    /// Portable scalar reference implementation. Always available.
    Scalar,
    /// x86_64 SSE4.1.
    Sse41,
    /// x86_64 AVX2.
    Avx2,
    /// aarch64 Advanced SIMD.
    Neon,
}

impl SimdLevel {
    /// Human-readable name, used in benchmark output and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse41 => "sse4.1",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Neon => "neon",
        }
    }

    /// Whether the current process can actually execute this level.
    pub fn is_supported(self) -> bool {
        match self {
            SimdLevel::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse41 => is_x86_feature_detected!("sse4.1"),
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => true,
            #[cfg(not(target_arch = "x86_64"))]
            SimdLevel::Sse41 | SimdLevel::Avx2 => false,
            #[cfg(not(target_arch = "aarch64"))]
            SimdLevel::Neon => false,
        }
    }
}

/// The best backend this host supports, detected once and cached.
///
/// Detection is deliberately done at run time rather than through
/// `#[cfg(target_feature)]` so a single binary built for baseline x86_64
/// still uses AVX2 where it exists, and safely falls back to the scalar
/// kernels where it does not.
pub fn detected_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        for level in [SimdLevel::Avx2, SimdLevel::Sse41, SimdLevel::Neon] {
            if level.is_supported() {
                return level;
            }
        }
        SimdLevel::Scalar
    })
}

/// Every backend this host can execute, best first. Always ends with
/// [`SimdLevel::Scalar`].
pub fn available_levels() -> Vec<SimdLevel> {
    let mut levels: Vec<SimdLevel> = [SimdLevel::Avx2, SimdLevel::Sse41, SimdLevel::Neon]
        .into_iter()
        .filter(|level| level.is_supported())
        .collect();
    levels.push(SimdLevel::Scalar);
    levels
}

/// A borrowed 8-bit reference plane that motion vectors index into.
#[derive(Clone, Copy, Debug)]
pub struct RefPlane<'a> {
    /// Sample storage, `stride * height` bytes or longer.
    pub data: &'a [u8],
    /// Visible width in samples.
    pub width: usize,
    /// Visible height in samples.
    pub height: usize,
    /// Distance between consecutive rows in `data`.
    pub stride: usize,
}

impl<'a> RefPlane<'a> {
    /// Builds a plane whose stride equals its width.
    ///
    /// # Panics
    ///
    /// Panics if `data` is too short for `width * height`.
    pub fn new(data: &'a [u8], width: usize, height: usize) -> RefPlane<'a> {
        assert!(data.len() >= width * height, "reference plane too short");
        RefPlane {
            data,
            width,
            height,
            stride: width,
        }
    }
}

/// Reusable motion-compensation working state.
///
/// Sub-pel prediction needs an edge-extended copy of the reference window
/// and a `(height + 7)`-row intermediate buffer for the horizontal pass.
/// Holding both here lets a decoder predict thousands of blocks per frame
/// without allocating per block.
#[derive(Clone, Debug)]
pub struct McContext {
    level: SimdLevel,
    window: Vec<u8>,
    intermediate: Vec<i16>,
}

impl Default for McContext {
    fn default() -> Self {
        McContext::new()
    }
}

impl McContext {
    /// Creates a context bound to the best backend this host supports.
    pub fn new() -> McContext {
        McContext::with_level(detected_level())
    }

    /// Creates a context pinned to `level`, falling back to
    /// [`SimdLevel::Scalar`] when the host cannot execute it. Tests use this
    /// to exercise every backend; production code should use
    /// [`McContext::new`].
    pub fn with_level(level: SimdLevel) -> McContext {
        let level = if level.is_supported() {
            level
        } else {
            SimdLevel::Scalar
        };
        McContext {
            level,
            window: Vec::new(),
            intermediate: Vec::new(),
        }
    }

    /// The backend this context runs on.
    pub fn level(&self) -> SimdLevel {
        self.level
    }

    /// Predicts a `width` x `height` block into final 8-bit pixels.
    ///
    /// `x`/`y` are the block's full-pel position in `reference` (they may be
    /// negative or extend past the plane; samples outside are edge-extended
    /// exactly as the spec's clamped reference fetch requires), and
    /// `subpel_x`/`subpel_y` are 1/16-pel phases in `0..16`.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is too short for `height` rows of `dst_stride`, if a
    /// sub-pel phase is out of range, or if the block is empty.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_single(
        &mut self,
        reference: RefPlane<'_>,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        subpel_x: usize,
        subpel_y: usize,
        filter: InterpFilter,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        self.prepare(reference, x, y, width, height, subpel_x, subpel_y, filter);
        assert!(
            dst.len() >= (height - 1) * dst_stride + width,
            "dst too short"
        );
        let level = self.level;
        vertical_u8(
            level,
            &self.intermediate,
            width,
            height,
            filter.taps(height, subpel_y),
            INTER_ROUND1_SINGLE,
            dst,
            dst_stride,
        );
    }

    /// Predicts a `width` x `height` block into intermediate compound
    /// samples at 16x pixel scale, ready for one of the blend kernels.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`McContext::predict_single`].
    #[allow(clippy::too_many_arguments)]
    pub fn predict_compound(
        &mut self,
        reference: RefPlane<'_>,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        subpel_x: usize,
        subpel_y: usize,
        filter: InterpFilter,
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        self.prepare(reference, x, y, width, height, subpel_x, subpel_y, filter);
        assert!(
            dst.len() >= (height - 1) * dst_stride + width,
            "dst too short"
        );
        let level = self.level;
        vertical_i16(
            level,
            &self.intermediate,
            width,
            height,
            filter.taps(height, subpel_y),
            INTER_ROUND1_COMPOUND,
            dst,
            dst_stride,
        );
    }

    /// Gathers the edge-extended reference window and runs the horizontal
    /// pass into `self.intermediate`, leaving `(height + 7)` rows of `width`
    /// samples for the vertical pass.
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &mut self,
        reference: RefPlane<'_>,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        subpel_x: usize,
        subpel_y: usize,
        filter: InterpFilter,
    ) {
        assert!(width > 0 && height > 0, "empty prediction block");
        assert!(
            subpel_x < SUBPEL_SHIFTS && subpel_y < SUBPEL_SHIFTS,
            "sub-pel phase out of range"
        );
        let rows = height + FILTER_MARGIN;
        let window_stride = width + FILTER_MARGIN;
        self.gather(reference, x, y, window_stride, rows);
        self.intermediate.clear();
        self.intermediate.resize(width * rows, 0);
        horizontal(
            self.level,
            &self.window,
            window_stride,
            width,
            rows,
            filter.taps(width, subpel_x),
            &mut self.intermediate,
        );
    }

    /// Copies the `(width + 7)` x `(height + 7)` reference window starting
    /// at `(x - 3, y - 3)` into `self.window`, replicating edge samples for
    /// positions outside the plane.
    fn gather(&mut self, reference: RefPlane<'_>, x: i32, y: i32, stride: usize, rows: usize) {
        self.window.clear();
        self.window.resize(stride * rows, 0);
        let left = x - FILTER_LEFT as i32;
        let top = y - FILTER_LEFT as i32;
        let plane_width = reference.width as i32;
        let plane_height = reference.height as i32;
        let interior = left >= 0 && left + stride as i32 <= plane_width;
        for row in 0..rows {
            let source_y = (top + row as i32).clamp(0, plane_height - 1) as usize;
            let source_row = &reference.data[source_y * reference.stride..][..reference.width];
            let dest = &mut self.window[row * stride..][..stride];
            if interior {
                dest.copy_from_slice(&source_row[left as usize..][..stride]);
            } else {
                for (column, sample) in dest.iter_mut().enumerate() {
                    let source_x = (left + column as i32).clamp(0, plane_width - 1) as usize;
                    *sample = source_row[source_x];
                }
            }
        }
    }
}

/// Distance-weighted compound prediction weights (spec §7.11.3.15).
///
/// `d0`/`d1` are the absolute order-hint distances from the current frame to
/// the two references backing `pred0` and `pred1`. The returned weights sum
/// to 16 and are fed to [`blend_distance`].
pub fn distance_weights(d0: u32, d1: u32) -> (i16, i16) {
    // Spec `Quant_Dist_Weight` / `Quant_Dist_Lookup`: walk the ratio table
    // until the observed distance ratio crosses a bucket boundary, keeping
    // the comparison one-sided by indexing both tables with `order`.
    const RATIOS: [[u32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE + 1]];
    const LOOKUP: [[i16; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];
    let d0 = d0.min(MAX_FRAME_DISTANCE);
    let d1 = d1.min(MAX_FRAME_DISTANCE);
    let order = usize::from(d0 <= d1);
    if d0 == 0 || d1 == 0 {
        return (LOOKUP[3][1 - order], LOOKUP[3][order]);
    }
    let mut bucket = 3;
    for (index, ratio) in RATIOS.iter().enumerate().take(3) {
        let scaled0 = u64::from(d0) * u64::from(ratio[order]);
        let scaled1 = u64::from(d1) * u64::from(ratio[1 - order]);
        if (d0 > d1 && scaled0 < scaled1) || (d0 <= d1 && scaled0 > scaled1) {
            bucket = index;
            break;
        }
    }
    (LOOKUP[bucket][1 - order], LOOKUP[bucket][order])
}

/// Blends two compound predictions with a simple average (spec §7.11.3.1,
/// `COMPOUND_AVERAGE`).
///
/// # Panics
///
/// Panics if any buffer is too short for the block.
#[allow(clippy::too_many_arguments)]
pub fn blend_average(
    level: SimdLevel,
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    blend_weighted(
        level, pred0, stride0, pred1, stride1, 1, 1, width, height, dst, dst_stride,
    );
}

/// Blends two compound predictions with distance weights summing to 16
/// (spec §7.11.3.15, `COMPOUND_DISTANCE`).
///
/// # Panics
///
/// Panics if the weights do not sum to 16, or if any buffer is too short.
#[allow(clippy::too_many_arguments)]
pub fn blend_distance(
    level: SimdLevel,
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    weight0: i16,
    weight1: i16,
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    assert_eq!(weight0 + weight1, 16, "distance weights must sum to 16");
    blend_weighted(
        level, pred0, stride0, pred1, stride1, weight0, weight1, width, height, dst, dst_stride,
    );
}

/// Shared implementation for the two constant-weight compound blends.
///
/// The weights sum to a power of two, so the post-blend shift is
/// `InterPostRound + log2(weight0 + weight1)`.
#[allow(clippy::too_many_arguments)]
fn blend_weighted(
    level: SimdLevel,
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    weight0: i16,
    weight1: i16,
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let total = weight0 as u32 + weight1 as u32;
    debug_assert!(
        total.is_power_of_two(),
        "blend weights must sum to a power of two"
    );
    let shift = INTER_POST_ROUND + total.trailing_zeros();
    check_block(pred0, stride0, width, height);
    check_block(pred1, stride1, width, height);
    assert!(
        dst.len() >= (height - 1) * dst_stride + width,
        "dst too short"
    );
    let params = BlendParams {
        weight0,
        weight1,
        shift,
        width,
        height,
    };
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            x86::blend_weighted_avx2(pred0, stride0, pred1, stride1, params, dst, dst_stride);
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe {
            x86::blend_weighted_sse41(pred0, stride0, pred1, stride1, params, dst, dst_stride);
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            neon::blend_weighted_neon(pred0, stride0, pred1, stride1, params, dst, dst_stride);
        },
        _ => blend_weighted_scalar(pred0, stride0, pred1, stride1, params, dst, dst_stride),
    }
}

/// Blends two compound predictions through a per-sample 0..=64 mask (spec
/// §7.11.3.14). Both wedge and difference-weighted compound modes reduce to
/// this kernel once their mask is built.
///
/// # Panics
///
/// Panics if any buffer is too short for the block.
#[allow(clippy::too_many_arguments)]
pub fn blend_mask(
    level: SimdLevel,
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    mask: &[u8],
    mask_stride: usize,
    width: usize,
    height: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    check_block(pred0, stride0, width, height);
    check_block(pred1, stride1, width, height);
    assert!(
        mask.len() >= (height - 1) * mask_stride + width,
        "mask too short"
    );
    assert!(
        dst.len() >= (height - 1) * dst_stride + width,
        "dst too short"
    );
    // `mask * pred0 + (64 - mask) * pred1` carries six extra fractional
    // bits on top of the compound scale.
    let shift = INTER_POST_ROUND + 6;
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            x86::blend_mask_avx2(
                pred0,
                stride0,
                pred1,
                stride1,
                mask,
                mask_stride,
                width,
                height,
                shift,
                dst,
                dst_stride,
            );
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe {
            x86::blend_mask_sse41(
                pred0,
                stride0,
                pred1,
                stride1,
                mask,
                mask_stride,
                width,
                height,
                shift,
                dst,
                dst_stride,
            );
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            neon::blend_mask_neon(
                pred0,
                stride0,
                pred1,
                stride1,
                mask,
                mask_stride,
                width,
                height,
                shift,
                dst,
                dst_stride,
            );
        },
        _ => blend_mask_scalar(
            pred0,
            stride0,
            pred1,
            stride1,
            mask,
            mask_stride,
            width,
            height,
            shift,
            dst,
            dst_stride,
        ),
    }
}

/// Builds the difference-weighted compound mask (spec §7.11.3.12) from two
/// compound predictions. `invert` corresponds to the coded `mask_type`.
///
/// # Panics
///
/// Panics if any buffer is too short for the block.
#[allow(clippy::too_many_arguments)]
pub fn build_difference_mask(
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    width: usize,
    height: usize,
    invert: bool,
    mask: &mut [u8],
    mask_stride: usize,
) {
    check_block(pred0, stride0, width, height);
    check_block(pred1, stride1, width, height);
    assert!(
        mask.len() >= (height - 1) * mask_stride + width,
        "mask too short"
    );
    for row in 0..height {
        let source0 = &pred0[row * stride0..][..width];
        let source1 = &pred1[row * stride1..][..width];
        let dest = &mut mask[row * mask_stride..][..width];
        for ((sample0, sample1), out) in source0.iter().zip(source1).zip(dest) {
            let difference = round2(
                (i32::from(*sample0) - i32::from(*sample1)).abs(),
                INTER_POST_ROUND,
            );
            let alpha = (DIFF_MASK_BASE + difference / DIFF_MASK_FACTOR)
                .clamp(0, i32::from(MAX_MASK_ALPHA));
            *out = if invert {
                MAX_MASK_ALPHA - alpha as u8
            } else {
                alpha as u8
            };
        }
    }
}

/// Verifies a compound prediction buffer covers the block.
fn check_block(pred: &[i16], stride: usize, width: usize, height: usize) {
    assert!(width > 0 && height > 0, "empty prediction block");
    assert!(
        pred.len() >= (height - 1) * stride + width,
        "prediction buffer too short"
    );
}

/// Constant-weight blend parameters, bundled to keep the kernel signatures
/// within reason.
#[derive(Clone, Copy, Debug)]
struct BlendParams {
    weight0: i16,
    weight1: i16,
    shift: u32,
    width: usize,
    height: usize,
}

/// Horizontal pass: convolves `rows` rows of the edge-extended window into
/// `width`-sample rows of `InterRound0`-rounded intermediates.
#[allow(clippy::too_many_arguments)]
fn horizontal(
    level: SimdLevel,
    window: &[u8],
    window_stride: usize,
    width: usize,
    rows: usize,
    taps: &[i16; 8],
    intermediate: &mut [i16],
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            x86::horizontal_avx2(window, window_stride, width, rows, taps, intermediate);
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe {
            x86::horizontal_sse41(window, window_stride, width, rows, taps, intermediate);
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            neon::horizontal_neon(window, window_stride, width, rows, taps, intermediate);
        },
        _ => horizontal_scalar(window, window_stride, width, rows, taps, intermediate),
    }
}

/// Vertical pass producing final 8-bit pixels.
#[allow(clippy::too_many_arguments)]
fn vertical_u8(
    level: SimdLevel,
    intermediate: &[i16],
    width: usize,
    height: usize,
    taps: &[i16; 8],
    shift: u32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            x86::vertical_u8_avx2(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe {
            x86::vertical_u8_sse41(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            neon::vertical_u8_neon(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        _ => vertical_u8_scalar(intermediate, width, height, taps, shift, dst, dst_stride),
    }
}

/// Vertical pass producing 16x-scale compound samples.
#[allow(clippy::too_many_arguments)]
fn vertical_i16(
    level: SimdLevel,
    intermediate: &[i16],
    width: usize,
    height: usize,
    taps: &[i16; 8],
    shift: u32,
    dst: &mut [i16],
    dst_stride: usize,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            x86::vertical_i16_avx2(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe {
            x86::vertical_i16_sse41(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            neon::vertical_i16_neon(intermediate, width, height, taps, shift, dst, dst_stride);
        },
        _ => vertical_i16_scalar(intermediate, width, height, taps, shift, dst, dst_stride),
    }
}

/// Scalar reference for the horizontal pass. Every vectorized backend must
/// reproduce this bit for bit.
fn horizontal_scalar(
    window: &[u8],
    window_stride: usize,
    width: usize,
    rows: usize,
    taps: &[i16; 8],
    intermediate: &mut [i16],
) {
    for row in 0..rows {
        let source = &window[row * window_stride..][..width + FILTER_MARGIN];
        let dest = &mut intermediate[row * width..][..width];
        for (column, out) in dest.iter_mut().enumerate() {
            let mut sum = 0i32;
            for (tap, coefficient) in taps.iter().enumerate() {
                sum += i32::from(*coefficient) * i32::from(source[column + tap]);
            }
            *out = round2(sum, INTER_ROUND0) as i16;
        }
    }
}

/// Scalar reference for the vertical pass into 8-bit pixels.
#[allow(clippy::too_many_arguments)]
fn vertical_u8_scalar(
    intermediate: &[i16],
    width: usize,
    height: usize,
    taps: &[i16; 8],
    shift: u32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    for row in 0..height {
        for column in 0..width {
            let mut sum = 0i32;
            for (tap, coefficient) in taps.iter().enumerate() {
                sum +=
                    i32::from(*coefficient) * i32::from(intermediate[(row + tap) * width + column]);
            }
            dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
        }
    }
}

/// Scalar reference for the vertical pass into compound samples.
#[allow(clippy::too_many_arguments)]
fn vertical_i16_scalar(
    intermediate: &[i16],
    width: usize,
    height: usize,
    taps: &[i16; 8],
    shift: u32,
    dst: &mut [i16],
    dst_stride: usize,
) {
    for row in 0..height {
        for column in 0..width {
            let mut sum = 0i32;
            for (tap, coefficient) in taps.iter().enumerate() {
                sum +=
                    i32::from(*coefficient) * i32::from(intermediate[(row + tap) * width + column]);
            }
            dst[row * dst_stride + column] = round2(sum, shift) as i16;
        }
    }
}

/// Scalar reference for the constant-weight compound blends.
fn blend_weighted_scalar(
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    params: BlendParams,
    dst: &mut [u8],
    dst_stride: usize,
) {
    for row in 0..params.height {
        let source0 = &pred0[row * stride0..][..params.width];
        let source1 = &pred1[row * stride1..][..params.width];
        let dest = &mut dst[row * dst_stride..][..params.width];
        for ((sample0, sample1), out) in source0.iter().zip(source1).zip(dest) {
            let sum = i32::from(params.weight0) * i32::from(*sample0)
                + i32::from(params.weight1) * i32::from(*sample1);
            *out = clip_pixel(round2(sum, params.shift));
        }
    }
}

/// Scalar reference for the masked compound blend.
#[allow(clippy::too_many_arguments)]
fn blend_mask_scalar(
    pred0: &[i16],
    stride0: usize,
    pred1: &[i16],
    stride1: usize,
    mask: &[u8],
    mask_stride: usize,
    width: usize,
    height: usize,
    shift: u32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    for row in 0..height {
        let source0 = &pred0[row * stride0..][..width];
        let source1 = &pred1[row * stride1..][..width];
        let alphas = &mask[row * mask_stride..][..width];
        let dest = &mut dst[row * dst_stride..][..width];
        for (((sample0, sample1), alpha), out) in source0.iter().zip(source1).zip(alphas).zip(dest)
        {
            let alpha = i32::from(*alpha);
            let sum = alpha * i32::from(*sample0)
                + (i32::from(MAX_MASK_ALPHA) - alpha) * i32::from(*sample1);
            *out = clip_pixel(round2(sum, shift));
        }
    }
}

/// SSE4.1 and AVX2 kernels.
///
/// Both backends keep the horizontal pass in 32-bit lanes (the tap sum can
/// exceed 16 bits) and the vertical pass in 16-bit lanes widened through the
/// `mullo`/`mulhi` pair, which reproduces the scalar `i32` products exactly.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{BlendParams, FILTER_MARGIN, INTER_ROUND0, MAX_MASK_ALPHA, clip_pixel, round2};
    use std::arch::x86_64::*;

    /// Broadcasts each tap into its own 128-bit 32-bit-lane vector.
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[target_feature(enable = "sse4.1")]
    unsafe fn taps_epi32(taps: &[i16; 8]) -> [__m128i; 8] {
        [
            _mm_set1_epi32(i32::from(taps[0])),
            _mm_set1_epi32(i32::from(taps[1])),
            _mm_set1_epi32(i32::from(taps[2])),
            _mm_set1_epi32(i32::from(taps[3])),
            _mm_set1_epi32(i32::from(taps[4])),
            _mm_set1_epi32(i32::from(taps[5])),
            _mm_set1_epi32(i32::from(taps[6])),
            _mm_set1_epi32(i32::from(taps[7])),
        ]
    }

    /// Broadcasts each tap into its own 256-bit 32-bit-lane vector.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn taps_epi32_avx2(taps: &[i16; 8]) -> [__m256i; 8] {
        [
            _mm256_set1_epi32(i32::from(taps[0])),
            _mm256_set1_epi32(i32::from(taps[1])),
            _mm256_set1_epi32(i32::from(taps[2])),
            _mm256_set1_epi32(i32::from(taps[3])),
            _mm256_set1_epi32(i32::from(taps[4])),
            _mm256_set1_epi32(i32::from(taps[5])),
            _mm256_set1_epi32(i32::from(taps[6])),
            _mm256_set1_epi32(i32::from(taps[7])),
        ]
    }

    /// Scalar remainder of the horizontal pass, for the columns a vector
    /// iteration cannot cover.
    fn horizontal_tail(
        source: &[u8],
        taps: &[i16; 8],
        dest: &mut [i16],
        first_column: usize,
        width: usize,
    ) {
        for column in first_column..width {
            let mut sum = 0i32;
            for (tap, coefficient) in taps.iter().enumerate() {
                sum += i32::from(*coefficient) * i32::from(source[column + tap]);
            }
            dest[column] = round2(sum, INTER_ROUND0) as i16;
        }
    }

    /// Horizontal pass, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1. `intermediate` must hold `rows * width` samples and
    /// `window` must hold `rows` rows of `width + 7` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn horizontal_sse41(
        window: &[u8],
        window_stride: usize,
        width: usize,
        rows: usize,
        taps: &[i16; 8],
        intermediate: &mut [i16],
    ) {
        unsafe {
            let coefficients = taps_epi32(taps);
            let rounding = _mm_set1_epi32(1 << (INTER_ROUND0 - 1));
            for row in 0..rows {
                let source = &window[row * window_stride..][..width + FILTER_MARGIN];
                let dest = &mut intermediate[row * width..][..width];
                let mut column = 0;
                while column + 8 <= width {
                    let mut low = rounding;
                    let mut high = rounding;
                    for (tap, coefficient) in coefficients.iter().enumerate() {
                        let bytes =
                            _mm_loadl_epi64(source.as_ptr().add(column + tap).cast::<__m128i>());
                        low = _mm_add_epi32(
                            low,
                            _mm_mullo_epi32(_mm_cvtepu8_epi32(bytes), *coefficient),
                        );
                        high = _mm_add_epi32(
                            high,
                            _mm_mullo_epi32(
                                _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4)),
                                *coefficient,
                            ),
                        );
                    }
                    let packed = _mm_packs_epi32(
                        _mm_srai_epi32::<{ INTER_ROUND0 as i32 }>(low),
                        _mm_srai_epi32::<{ INTER_ROUND0 as i32 }>(high),
                    );
                    _mm_storeu_si128(dest.as_mut_ptr().add(column).cast::<__m128i>(), packed);
                    column += 8;
                }
                horizontal_tail(source, taps, dest, column, width);
            }
        }
    }

    /// Horizontal pass, eight samples per iteration in 256-bit lanes.
    ///
    /// # Safety
    ///
    /// Requires AVX2, with the same buffer requirements as
    /// [`horizontal_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn horizontal_avx2(
        window: &[u8],
        window_stride: usize,
        width: usize,
        rows: usize,
        taps: &[i16; 8],
        intermediate: &mut [i16],
    ) {
        unsafe {
            let coefficients = taps_epi32_avx2(taps);
            let rounding = _mm256_set1_epi32(1 << (INTER_ROUND0 - 1));
            for row in 0..rows {
                let source = &window[row * window_stride..][..width + FILTER_MARGIN];
                let dest = &mut intermediate[row * width..][..width];
                let mut column = 0;
                while column + 8 <= width {
                    let mut accumulator = rounding;
                    for (tap, coefficient) in coefficients.iter().enumerate() {
                        let bytes =
                            _mm_loadl_epi64(source.as_ptr().add(column + tap).cast::<__m128i>());
                        accumulator = _mm256_add_epi32(
                            accumulator,
                            _mm256_mullo_epi32(_mm256_cvtepu8_epi32(bytes), *coefficient),
                        );
                    }
                    let shifted = _mm256_srai_epi32::<{ INTER_ROUND0 as i32 }>(accumulator);
                    let packed = _mm_packs_epi32(
                        _mm256_castsi256_si128(shifted),
                        _mm256_extracti128_si256(shifted, 1),
                    );
                    _mm_storeu_si128(dest.as_mut_ptr().add(column).cast::<__m128i>(), packed);
                    column += 8;
                }
                horizontal_tail(source, taps, dest, column, width);
            }
        }
    }

    /// Widens eight 16-bit products into two 32-bit accumulators.
    ///
    /// # Safety
    ///
    /// Requires SSE2.
    #[target_feature(enable = "sse4.1")]
    unsafe fn accumulate_sse41(
        low: __m128i,
        high: __m128i,
        values: __m128i,
        coefficient: __m128i,
    ) -> (__m128i, __m128i) {
        let products_low = _mm_mullo_epi16(values, coefficient);
        let products_high = _mm_mulhi_epi16(values, coefficient);
        (
            _mm_add_epi32(low, _mm_unpacklo_epi16(products_low, products_high)),
            _mm_add_epi32(high, _mm_unpackhi_epi16(products_low, products_high)),
        )
    }

    /// Widens sixteen 16-bit products into two 32-bit accumulators.
    ///
    /// # Safety
    ///
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_avx2(
        low: __m256i,
        high: __m256i,
        values: __m256i,
        coefficient: __m256i,
    ) -> (__m256i, __m256i) {
        let products_low = _mm256_mullo_epi16(values, coefficient);
        let products_high = _mm256_mulhi_epi16(values, coefficient);
        (
            _mm256_add_epi32(low, _mm256_unpacklo_epi16(products_low, products_high)),
            _mm256_add_epi32(high, _mm256_unpackhi_epi16(products_low, products_high)),
        )
    }

    /// Vertical pass into 8-bit pixels, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1. `intermediate` must hold `(height + 7) * width`
    /// samples and `dst` must cover the block.
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_u8_sse41(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let rounding = _mm_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let (mut low, mut high) = (rounding, rounding);
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values = _mm_loadu_si128(
                            intermediate
                                .as_ptr()
                                .add((row + tap) * width + column)
                                .cast::<__m128i>(),
                        );
                        let (a, b) =
                            accumulate_sse41(low, high, values, _mm_set1_epi16(*coefficient));
                        low = a;
                        high = b;
                    }
                    let packed =
                        _mm_packs_epi32(_mm_sra_epi32(low, down), _mm_sra_epi32(high, down));
                    _mm_storel_epi64(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_packus_epi16(packed, packed),
                    );
                    column += 8;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }

    /// Vertical pass into compound samples, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1, with the same buffer requirements as
    /// [`vertical_u8_sse41`].
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_i16_sse41(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        unsafe {
            let rounding = _mm_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let (mut low, mut high) = (rounding, rounding);
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values = _mm_loadu_si128(
                            intermediate
                                .as_ptr()
                                .add((row + tap) * width + column)
                                .cast::<__m128i>(),
                        );
                        let (a, b) =
                            accumulate_sse41(low, high, values, _mm_set1_epi16(*coefficient));
                        low = a;
                        high = b;
                    }
                    let packed =
                        _mm_packs_epi32(_mm_sra_epi32(low, down), _mm_sra_epi32(high, down));
                    _mm_storeu_si128(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        packed,
                    );
                    column += 8;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = round2(sum, shift) as i16;
                    column += 1;
                }
            }
        }
    }

    /// Vertical pass into 8-bit pixels, sixteen samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires AVX2, with the same buffer requirements as
    /// [`vertical_u8_sse41`].
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_u8_avx2(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let rounding = _mm256_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 16 <= width {
                    let (mut low, mut high) = (rounding, rounding);
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values = _mm256_loadu_si256(
                            intermediate
                                .as_ptr()
                                .add((row + tap) * width + column)
                                .cast::<__m256i>(),
                        );
                        let (a, b) =
                            accumulate_avx2(low, high, values, _mm256_set1_epi16(*coefficient));
                        low = a;
                        high = b;
                    }
                    let packed = _mm256_packs_epi32(
                        _mm256_sra_epi32(low, down),
                        _mm256_sra_epi32(high, down),
                    );
                    let bytes = _mm256_packus_epi16(packed, packed);
                    _mm_storeu_si128(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_unpacklo_epi64(
                            _mm256_castsi256_si128(bytes),
                            _mm256_extracti128_si256(bytes, 1),
                        ),
                    );
                    column += 16;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }

    /// Vertical pass into compound samples, sixteen samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires AVX2, with the same buffer requirements as
    /// [`vertical_u8_sse41`].
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_i16_avx2(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        unsafe {
            let rounding = _mm256_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 16 <= width {
                    let (mut low, mut high) = (rounding, rounding);
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values = _mm256_loadu_si256(
                            intermediate
                                .as_ptr()
                                .add((row + tap) * width + column)
                                .cast::<__m256i>(),
                        );
                        let (a, b) =
                            accumulate_avx2(low, high, values, _mm256_set1_epi16(*coefficient));
                        low = a;
                        high = b;
                    }
                    let packed = _mm256_packs_epi32(
                        _mm256_sra_epi32(low, down),
                        _mm256_sra_epi32(high, down),
                    );
                    _mm256_storeu_si256(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m256i>(),
                        packed,
                    );
                    column += 16;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = round2(sum, shift) as i16;
                    column += 1;
                }
            }
        }
    }

    /// Constant-weight compound blend, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1. All buffers must cover the block described by
    /// `params`.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn blend_weighted_sse41(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        params: BlendParams,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let weight0 = _mm_set1_epi16(params.weight0);
            let weight1 = _mm_set1_epi16(params.weight1);
            let rounding = _mm_set1_epi32(1 << (params.shift - 1));
            let down = _mm_cvtsi32_si128(params.shift as i32);
            for row in 0..params.height {
                let mut column = 0;
                while column + 8 <= params.width {
                    let first = _mm_loadu_si128(
                        pred0.as_ptr().add(row * stride0 + column).cast::<__m128i>(),
                    );
                    let second = _mm_loadu_si128(
                        pred1.as_ptr().add(row * stride1 + column).cast::<__m128i>(),
                    );
                    let (low, high) = accumulate_sse41(rounding, rounding, first, weight0);
                    let (low, high) = accumulate_sse41(low, high, second, weight1);
                    let packed =
                        _mm_packs_epi32(_mm_sra_epi32(low, down), _mm_sra_epi32(high, down));
                    _mm_storel_epi64(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_packus_epi16(packed, packed),
                    );
                    column += 8;
                }
                while column < params.width {
                    let sum = i32::from(params.weight0) * i32::from(pred0[row * stride0 + column])
                        + i32::from(params.weight1) * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, params.shift));
                    column += 1;
                }
            }
        }
    }

    /// Constant-weight compound blend, sixteen samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires AVX2, with the same buffer requirements as
    /// [`blend_weighted_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn blend_weighted_avx2(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        params: BlendParams,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let weight0 = _mm256_set1_epi16(params.weight0);
            let weight1 = _mm256_set1_epi16(params.weight1);
            let rounding = _mm256_set1_epi32(1 << (params.shift - 1));
            let down = _mm_cvtsi32_si128(params.shift as i32);
            for row in 0..params.height {
                let mut column = 0;
                while column + 16 <= params.width {
                    let first = _mm256_loadu_si256(
                        pred0.as_ptr().add(row * stride0 + column).cast::<__m256i>(),
                    );
                    let second = _mm256_loadu_si256(
                        pred1.as_ptr().add(row * stride1 + column).cast::<__m256i>(),
                    );
                    let (low, high) = accumulate_avx2(rounding, rounding, first, weight0);
                    let (low, high) = accumulate_avx2(low, high, second, weight1);
                    let packed = _mm256_packs_epi32(
                        _mm256_sra_epi32(low, down),
                        _mm256_sra_epi32(high, down),
                    );
                    let bytes = _mm256_packus_epi16(packed, packed);
                    _mm_storeu_si128(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_unpacklo_epi64(
                            _mm256_castsi256_si128(bytes),
                            _mm256_extracti128_si256(bytes, 1),
                        ),
                    );
                    column += 16;
                }
                while column < params.width {
                    let sum = i32::from(params.weight0) * i32::from(pred0[row * stride0 + column])
                        + i32::from(params.weight1) * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, params.shift));
                    column += 1;
                }
            }
        }
    }

    /// Masked compound blend, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires SSE4.1. All buffers must cover the block.
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn blend_mask_sse41(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        mask: &[u8],
        mask_stride: usize,
        width: usize,
        height: usize,
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let full = _mm_set1_epi16(i16::from(MAX_MASK_ALPHA));
            let rounding = _mm_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let first = _mm_loadu_si128(
                        pred0.as_ptr().add(row * stride0 + column).cast::<__m128i>(),
                    );
                    let second = _mm_loadu_si128(
                        pred1.as_ptr().add(row * stride1 + column).cast::<__m128i>(),
                    );
                    let alpha = _mm_cvtepu8_epi16(_mm_loadl_epi64(
                        mask.as_ptr()
                            .add(row * mask_stride + column)
                            .cast::<__m128i>(),
                    ));
                    let inverse = _mm_sub_epi16(full, alpha);
                    let (low, high) = accumulate_sse41(rounding, rounding, first, alpha);
                    let (low, high) = accumulate_sse41(low, high, second, inverse);
                    let packed =
                        _mm_packs_epi32(_mm_sra_epi32(low, down), _mm_sra_epi32(high, down));
                    _mm_storel_epi64(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_packus_epi16(packed, packed),
                    );
                    column += 8;
                }
                while column < width {
                    let alpha = i32::from(mask[row * mask_stride + column]);
                    let sum = alpha * i32::from(pred0[row * stride0 + column])
                        + (i32::from(MAX_MASK_ALPHA) - alpha)
                            * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }

    /// Masked compound blend, sixteen samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires AVX2, with the same buffer requirements as
    /// [`blend_mask_sse41`].
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn blend_mask_avx2(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        mask: &[u8],
        mask_stride: usize,
        width: usize,
        height: usize,
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let full = _mm256_set1_epi16(i16::from(MAX_MASK_ALPHA));
            let rounding = _mm256_set1_epi32(1 << (shift - 1));
            let down = _mm_cvtsi32_si128(shift as i32);
            for row in 0..height {
                let mut column = 0;
                while column + 16 <= width {
                    let first = _mm256_loadu_si256(
                        pred0.as_ptr().add(row * stride0 + column).cast::<__m256i>(),
                    );
                    let second = _mm256_loadu_si256(
                        pred1.as_ptr().add(row * stride1 + column).cast::<__m256i>(),
                    );
                    let alpha = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                        mask.as_ptr()
                            .add(row * mask_stride + column)
                            .cast::<__m128i>(),
                    ));
                    let inverse = _mm256_sub_epi16(full, alpha);
                    let (low, high) = accumulate_avx2(rounding, rounding, first, alpha);
                    let (low, high) = accumulate_avx2(low, high, second, inverse);
                    let packed = _mm256_packs_epi32(
                        _mm256_sra_epi32(low, down),
                        _mm256_sra_epi32(high, down),
                    );
                    let bytes = _mm256_packus_epi16(packed, packed);
                    _mm_storeu_si128(
                        dst.as_mut_ptr()
                            .add(row * dst_stride + column)
                            .cast::<__m128i>(),
                        _mm_unpacklo_epi64(
                            _mm256_castsi256_si128(bytes),
                            _mm256_extracti128_si256(bytes, 1),
                        ),
                    );
                    column += 16;
                }
                while column < width {
                    let alpha = i32::from(mask[row * mask_stride + column]);
                    let sum = alpha * i32::from(pred0[row * stride0 + column])
                        + (i32::from(MAX_MASK_ALPHA) - alpha)
                            * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }
}

/// NEON kernels.
///
/// `vmlal_n_s16` accumulates widening 16x16 -> 32-bit products directly, so
/// each kernel mirrors the scalar accumulation shape with two 32-bit
/// accumulators covering eight lanes.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{BlendParams, FILTER_MARGIN, INTER_ROUND0, MAX_MASK_ALPHA, clip_pixel, round2};
    use std::arch::aarch64::*;

    /// Horizontal pass, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires NEON. `intermediate` must hold `rows * width` samples and
    /// `window` must hold `rows` rows of `width + 7` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn horizontal_neon(
        window: &[u8],
        window_stride: usize,
        width: usize,
        rows: usize,
        taps: &[i16; 8],
        intermediate: &mut [i16],
    ) {
        unsafe {
            for row in 0..rows {
                let source = &window[row * window_stride..][..width + FILTER_MARGIN];
                let dest = &mut intermediate[row * width..][..width];
                let mut column = 0;
                while column + 8 <= width {
                    let mut low = vdupq_n_s32(1 << (INTER_ROUND0 - 1));
                    let mut high = low;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let bytes = vld1_u8(source.as_ptr().add(column + tap));
                        let widened = vreinterpretq_s16_u16(vmovl_u8(bytes));
                        low = vmlal_n_s16(low, vget_low_s16(widened), *coefficient);
                        high = vmlal_n_s16(high, vget_high_s16(widened), *coefficient);
                    }
                    let packed = vcombine_s16(
                        vqmovn_s32(vshrq_n_s32::<{ INTER_ROUND0 as i32 }>(low)),
                        vqmovn_s32(vshrq_n_s32::<{ INTER_ROUND0 as i32 }>(high)),
                    );
                    vst1q_s16(dest.as_mut_ptr().add(column), packed);
                    column += 8;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient) * i32::from(source[column + tap]);
                    }
                    dest[column] = round2(sum, INTER_ROUND0) as i16;
                    column += 1;
                }
            }
        }
    }

    /// Vertical pass into 8-bit pixels, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires NEON. `intermediate` must hold `(height + 7) * width`
    /// samples and `dst` must cover the block.
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_u8_neon(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let down = vdupq_n_s32(-(shift as i32));
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let mut low = vdupq_n_s32(1 << (shift - 1));
                    let mut high = low;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values =
                            vld1q_s16(intermediate.as_ptr().add((row + tap) * width + column));
                        low = vmlal_n_s16(low, vget_low_s16(values), *coefficient);
                        high = vmlal_n_s16(high, vget_high_s16(values), *coefficient);
                    }
                    let packed = vcombine_s16(
                        vqmovn_s32(vshlq_s32(low, down)),
                        vqmovn_s32(vshlq_s32(high, down)),
                    );
                    vst1_u8(
                        dst.as_mut_ptr().add(row * dst_stride + column),
                        vqmovun_s16(packed),
                    );
                    column += 8;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }

    /// Vertical pass into compound samples, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires NEON, with the same buffer requirements as
    /// [`vertical_u8_neon`].
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn vertical_i16_neon(
        intermediate: &[i16],
        width: usize,
        height: usize,
        taps: &[i16; 8],
        shift: u32,
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        unsafe {
            let down = vdupq_n_s32(-(shift as i32));
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let mut low = vdupq_n_s32(1 << (shift - 1));
                    let mut high = low;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        let values =
                            vld1q_s16(intermediate.as_ptr().add((row + tap) * width + column));
                        low = vmlal_n_s16(low, vget_low_s16(values), *coefficient);
                        high = vmlal_n_s16(high, vget_high_s16(values), *coefficient);
                    }
                    let packed = vcombine_s16(
                        vqmovn_s32(vshlq_s32(low, down)),
                        vqmovn_s32(vshlq_s32(high, down)),
                    );
                    vst1q_s16(dst.as_mut_ptr().add(row * dst_stride + column), packed);
                    column += 8;
                }
                while column < width {
                    let mut sum = 0i32;
                    for (tap, coefficient) in taps.iter().enumerate() {
                        sum += i32::from(*coefficient)
                            * i32::from(intermediate[(row + tap) * width + column]);
                    }
                    dst[row * dst_stride + column] = round2(sum, shift) as i16;
                    column += 1;
                }
            }
        }
    }

    /// Constant-weight compound blend, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires NEON. All buffers must cover the block described by
    /// `params`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn blend_weighted_neon(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        params: BlendParams,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let down = vdupq_n_s32(-(params.shift as i32));
            for row in 0..params.height {
                let mut column = 0;
                while column + 8 <= params.width {
                    let first = vld1q_s16(pred0.as_ptr().add(row * stride0 + column));
                    let second = vld1q_s16(pred1.as_ptr().add(row * stride1 + column));
                    let rounding = vdupq_n_s32(1 << (params.shift - 1));
                    let mut low = vmlal_n_s16(rounding, vget_low_s16(first), params.weight0);
                    let mut high = vmlal_n_s16(rounding, vget_high_s16(first), params.weight0);
                    low = vmlal_n_s16(low, vget_low_s16(second), params.weight1);
                    high = vmlal_n_s16(high, vget_high_s16(second), params.weight1);
                    let packed = vcombine_s16(
                        vqmovn_s32(vshlq_s32(low, down)),
                        vqmovn_s32(vshlq_s32(high, down)),
                    );
                    vst1_u8(
                        dst.as_mut_ptr().add(row * dst_stride + column),
                        vqmovun_s16(packed),
                    );
                    column += 8;
                }
                while column < params.width {
                    let sum = i32::from(params.weight0) * i32::from(pred0[row * stride0 + column])
                        + i32::from(params.weight1) * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, params.shift));
                    column += 1;
                }
            }
        }
    }

    /// Masked compound blend, eight samples per iteration.
    ///
    /// # Safety
    ///
    /// Requires NEON. All buffers must cover the block.
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn blend_mask_neon(
        pred0: &[i16],
        stride0: usize,
        pred1: &[i16],
        stride1: usize,
        mask: &[u8],
        mask_stride: usize,
        width: usize,
        height: usize,
        shift: u32,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        unsafe {
            let down = vdupq_n_s32(-(shift as i32));
            let full = vdupq_n_s16(i16::from(MAX_MASK_ALPHA));
            for row in 0..height {
                let mut column = 0;
                while column + 8 <= width {
                    let first = vld1q_s16(pred0.as_ptr().add(row * stride0 + column));
                    let second = vld1q_s16(pred1.as_ptr().add(row * stride1 + column));
                    let alpha = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(
                        mask.as_ptr().add(row * mask_stride + column),
                    )));
                    let inverse = vsubq_s16(full, alpha);
                    let rounding = vdupq_n_s32(1 << (shift - 1));
                    let mut low = vmlal_s16(rounding, vget_low_s16(first), vget_low_s16(alpha));
                    let mut high = vmlal_s16(rounding, vget_high_s16(first), vget_high_s16(alpha));
                    low = vmlal_s16(low, vget_low_s16(second), vget_low_s16(inverse));
                    high = vmlal_s16(high, vget_high_s16(second), vget_high_s16(inverse));
                    let packed = vcombine_s16(
                        vqmovn_s32(vshlq_s32(low, down)),
                        vqmovn_s32(vshlq_s32(high, down)),
                    );
                    vst1_u8(
                        dst.as_mut_ptr().add(row * dst_stride + column),
                        vqmovun_s16(packed),
                    );
                    column += 8;
                }
                while column < width {
                    let alpha = i32::from(mask[row * mask_stride + column]);
                    let sum = alpha * i32::from(pred0[row * stride0 + column])
                        + (i32::from(MAX_MASK_ALPHA) - alpha)
                            * i32::from(pred1[row * stride1 + column]);
                    dst[row * dst_stride + column] = clip_pixel(round2(sum, shift));
                    column += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG so the fixtures below are reproducible
    /// without pulling in a dependency.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed | 1)
        }

        fn next(&mut self) -> u32 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            (state >> 32) as u32
        }

        fn byte(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
    }

    /// A pseudo-random reference plane with a few flat and saturated
    /// regions, so kernels are exercised on both smooth and clipping input.
    fn reference_plane(width: usize, height: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        let mut plane = vec![0u8; width * height];
        for (index, sample) in plane.iter_mut().enumerate() {
            *sample = match index % 37 {
                0..=3 => 0,
                4..=7 => 255,
                _ => rng.byte(),
            };
        }
        plane
    }

    /// Block geometries covering vector-width multiples, sub-vector widths,
    /// and widths whose tail the vector loops cannot cover.
    const BLOCK_SIZES: [(usize, usize); 12] = [
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (12, 12),
        (16, 16),
        (16, 8),
        (20, 4),
        (32, 32),
        (33, 5),
        (64, 64),
        (7, 9),
    ];

    #[test]
    fn every_filter_phase_sums_to_unity() {
        for set in SUBPEL_FILTERS.iter() {
            for taps in set.iter() {
                let sum: i32 = taps.iter().map(|tap| i32::from(*tap)).sum();
                assert_eq!(sum, 1 << FILTER_BITS, "filter phase {taps:?} is not unity");
            }
        }
    }

    #[test]
    fn whole_pel_phase_is_a_pure_copy() {
        for set in SUBPEL_FILTERS.iter() {
            assert_eq!(set[0], [0, 0, 0, 1 << FILTER_BITS, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn narrow_blocks_select_the_four_tap_sets() {
        for filter in InterpFilter::ALL {
            for subpel in 1..SUBPEL_SHIFTS {
                let narrow = filter.taps(4, subpel);
                if filter != InterpFilter::Bilinear {
                    assert_eq!(narrow[0], 0, "narrow {filter:?} uses an eight-tap span");
                    assert_eq!(narrow[7], 0, "narrow {filter:?} uses an eight-tap span");
                }
            }
            // Wide blocks keep the full eight-tap span for the sharp filter.
            if filter == InterpFilter::Sharp {
                assert_ne!(filter.taps(8, 8)[0], 0);
            }
        }
    }

    #[test]
    fn flat_input_is_reproduced_exactly() {
        let plane = vec![137u8; 32 * 32];
        let reference = RefPlane::new(&plane, 32, 32);
        for level in available_levels() {
            let mut context = McContext::with_level(level);
            for filter in InterpFilter::ALL {
                for subpel_x in 0..SUBPEL_SHIFTS {
                    for subpel_y in 0..SUBPEL_SHIFTS {
                        let mut dst = vec![0u8; 16 * 16];
                        context.predict_single(
                            reference, 8, 8, 16, 16, subpel_x, subpel_y, filter, &mut dst, 16,
                        );
                        assert!(
                            dst.iter().all(|sample| *sample == 137),
                            "{} {filter:?} phase ({subpel_x},{subpel_y}) did not preserve a flat plane",
                            level.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn whole_pel_prediction_copies_the_reference() {
        let plane = reference_plane(48, 48, 7);
        let reference = RefPlane::new(&plane, 48, 48);
        for level in available_levels() {
            let mut context = McContext::with_level(level);
            for (width, height) in BLOCK_SIZES {
                if width > 48 || height > 48 {
                    continue;
                }
                let mut dst = vec![0u8; width * height];
                context.predict_single(
                    reference,
                    3,
                    5,
                    width,
                    height,
                    0,
                    0,
                    InterpFilter::Regular,
                    &mut dst,
                    width,
                );
                for row in 0..height {
                    let expected = &plane[(5 + row) * 48 + 3..][..width];
                    assert_eq!(&dst[row * width..][..width], expected, "{}", level.name());
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_the_scalar_single_prediction() {
        let plane = reference_plane(96, 72, 0x5eed);
        let reference = RefPlane::new(&plane, 96, 72);
        // Positions include blocks that hang off every edge, exercising the
        // clamped (edge-extended) reference fetch.
        let positions = [(10i32, 12i32), (-5, 3), (0, 0), (90, 68), (-9, -9), (33, 7)];
        let mut scalar = McContext::with_level(SimdLevel::Scalar);
        for level in available_levels() {
            if level == SimdLevel::Scalar {
                continue;
            }
            let mut context = McContext::with_level(level);
            assert_eq!(
                context.level(),
                level,
                "backend {} unavailable",
                level.name()
            );
            for (width, height) in BLOCK_SIZES {
                for (x, y) in positions {
                    for filter in InterpFilter::ALL {
                        for (subpel_x, subpel_y) in
                            [(0, 0), (1, 0), (0, 15), (5, 7), (8, 8), (15, 1)]
                        {
                            let mut expected = vec![0u8; width * height];
                            let mut actual = vec![0u8; width * height];
                            scalar.predict_single(
                                reference,
                                x,
                                y,
                                width,
                                height,
                                subpel_x,
                                subpel_y,
                                filter,
                                &mut expected,
                                width,
                            );
                            context.predict_single(
                                reference,
                                x,
                                y,
                                width,
                                height,
                                subpel_x,
                                subpel_y,
                                filter,
                                &mut actual,
                                width,
                            );
                            assert_eq!(
                                actual,
                                expected,
                                "{} differs at {width}x{height} ({x},{y}) {filter:?} phase ({subpel_x},{subpel_y})",
                                level.name()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_the_scalar_compound_prediction() {
        let plane = reference_plane(96, 72, 0xc0ffee);
        let reference = RefPlane::new(&plane, 96, 72);
        let mut scalar = McContext::with_level(SimdLevel::Scalar);
        for level in available_levels() {
            if level == SimdLevel::Scalar {
                continue;
            }
            let mut context = McContext::with_level(level);
            for (width, height) in BLOCK_SIZES {
                for filter in InterpFilter::ALL {
                    for (subpel_x, subpel_y) in [(0, 0), (3, 11), (15, 15), (9, 0)] {
                        let mut expected = vec![0i16; width * height];
                        let mut actual = vec![0i16; width * height];
                        scalar.predict_compound(
                            reference,
                            7,
                            6,
                            width,
                            height,
                            subpel_x,
                            subpel_y,
                            filter,
                            &mut expected,
                            width,
                        );
                        context.predict_compound(
                            reference,
                            7,
                            6,
                            width,
                            height,
                            subpel_x,
                            subpel_y,
                            filter,
                            &mut actual,
                            width,
                        );
                        assert_eq!(
                            actual,
                            expected,
                            "{} differs at {width}x{height} {filter:?} phase ({subpel_x},{subpel_y})",
                            level.name()
                        );
                    }
                }
            }
        }
    }

    /// Builds two compound predictions and a mask for the blend tests.
    fn compound_pair(width: usize, height: usize) -> (Vec<i16>, Vec<i16>, Vec<u8>) {
        let plane = reference_plane(128, 128, 0xbeef);
        let reference = RefPlane::new(&plane, 128, 128);
        let mut context = McContext::with_level(SimdLevel::Scalar);
        let mut first = vec![0i16; width * height];
        let mut second = vec![0i16; width * height];
        context.predict_compound(
            reference,
            5,
            9,
            width,
            height,
            5,
            7,
            InterpFilter::Sharp,
            &mut first,
            width,
        );
        context.predict_compound(
            reference,
            40,
            33,
            width,
            height,
            11,
            2,
            InterpFilter::Smooth,
            &mut second,
            width,
        );
        let mut mask = vec![0u8; width * height];
        build_difference_mask(
            &first, width, &second, width, width, height, false, &mut mask, width,
        );
        (first, second, mask)
    }

    #[test]
    fn every_backend_matches_the_scalar_compound_blends() {
        for (width, height) in BLOCK_SIZES {
            let (first, second, mask) = compound_pair(width, height);
            let mut expected = vec![0u8; width * height];
            blend_average(
                SimdLevel::Scalar,
                &first,
                width,
                &second,
                width,
                width,
                height,
                &mut expected,
                width,
            );
            for level in available_levels() {
                let mut actual = vec![0u8; width * height];
                blend_average(
                    level,
                    &first,
                    width,
                    &second,
                    width,
                    width,
                    height,
                    &mut actual,
                    width,
                );
                assert_eq!(actual, expected, "{} average blend", level.name());

                for (weight0, weight1) in [(9i16, 7i16), (11, 5), (12, 4), (13, 3), (7, 9)] {
                    let mut reference_blend = vec![0u8; width * height];
                    let mut candidate = vec![0u8; width * height];
                    blend_distance(
                        SimdLevel::Scalar,
                        &first,
                        width,
                        &second,
                        width,
                        weight0,
                        weight1,
                        width,
                        height,
                        &mut reference_blend,
                        width,
                    );
                    blend_distance(
                        level,
                        &first,
                        width,
                        &second,
                        width,
                        weight0,
                        weight1,
                        width,
                        height,
                        &mut candidate,
                        width,
                    );
                    assert_eq!(
                        candidate,
                        reference_blend,
                        "{} distance blend {weight0}/{weight1}",
                        level.name()
                    );
                }

                let mut reference_blend = vec![0u8; width * height];
                let mut candidate = vec![0u8; width * height];
                blend_mask(
                    SimdLevel::Scalar,
                    &first,
                    width,
                    &second,
                    width,
                    &mask,
                    width,
                    width,
                    height,
                    &mut reference_blend,
                    width,
                );
                blend_mask(
                    level,
                    &first,
                    width,
                    &second,
                    width,
                    &mask,
                    width,
                    width,
                    height,
                    &mut candidate,
                    width,
                );
                assert_eq!(candidate, reference_blend, "{} masked blend", level.name());
            }
        }
    }

    #[test]
    fn average_blend_matches_the_naive_whole_pel_average() {
        // At whole-pel phases the compound path carries the reference
        // samples at 16x scale, so the average blend must agree with the
        // plain `(a + b + 1) >> 1` a scalar decoder would compute.
        let plane = reference_plane(64, 64, 0x1234);
        let reference = RefPlane::new(&plane, 64, 64);
        let mut context = McContext::new();
        let (width, height) = (16, 16);
        let mut first = vec![0i16; width * height];
        let mut second = vec![0i16; width * height];
        context.predict_compound(
            reference,
            4,
            4,
            width,
            height,
            0,
            0,
            InterpFilter::Regular,
            &mut first,
            width,
        );
        context.predict_compound(
            reference,
            20,
            30,
            width,
            height,
            0,
            0,
            InterpFilter::Regular,
            &mut second,
            width,
        );
        let mut blended = vec![0u8; width * height];
        blend_average(
            context.level(),
            &first,
            width,
            &second,
            width,
            width,
            height,
            &mut blended,
            width,
        );
        for row in 0..height {
            for column in 0..width {
                let a = u16::from(plane[(4 + row) * 64 + 4 + column]);
                let b = u16::from(plane[(30 + row) * 64 + 20 + column]);
                assert_eq!(blended[row * width + column], ((a + b + 1) >> 1) as u8);
            }
        }
    }

    #[test]
    fn difference_mask_stays_in_range_and_inverts() {
        let (width, height) = (16, 16);
        let (first, second, mask) = compound_pair(width, height);
        let mut inverted = vec![0u8; width * height];
        build_difference_mask(
            &first,
            width,
            &second,
            width,
            width,
            height,
            true,
            &mut inverted,
            width,
        );
        for (alpha, complement) in mask.iter().zip(&inverted) {
            assert!(*alpha <= MAX_MASK_ALPHA);
            assert_eq!(*alpha + *complement, MAX_MASK_ALPHA);
        }
        assert!(mask.iter().any(|alpha| *alpha > DIFF_MASK_BASE as u8));
    }

    #[test]
    fn masked_blend_at_the_extremes_selects_one_prediction() {
        let (width, height) = (16, 16);
        let (first, second, _) = compound_pair(width, height);
        for (alpha, expected_source) in [(MAX_MASK_ALPHA, &first), (0, &second)] {
            let mask = vec![alpha; width * height];
            let mut blended = vec![0u8; width * height];
            blend_mask(
                detected_level(),
                &first,
                width,
                &second,
                width,
                &mask,
                width,
                width,
                height,
                &mut blended,
                width,
            );
            for (sample, source) in blended.iter().zip(expected_source) {
                assert_eq!(
                    *sample,
                    clip_pixel(round2(i32::from(*source), INTER_POST_ROUND))
                );
            }
        }
    }

    #[test]
    fn distance_weights_sum_to_sixteen_and_mirror() {
        for d0 in 0..12u32 {
            for d1 in 0..12u32 {
                let (weight0, weight1) = distance_weights(d0, d1);
                assert_eq!(weight0 + weight1, 16, "weights for ({d0},{d1})");
                if d0 != d1 {
                    let (mirror0, mirror1) = distance_weights(d1, d0);
                    assert_eq!((mirror1, mirror0), (weight0, weight1));
                }
                if d0 > 0 && d1 > 0 && d0 < d1 {
                    assert!(
                        weight0 >= weight1,
                        "the nearer reference must not lose weight at ({d0},{d1})"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn simd_backends_outrun_the_scalar_kernels() {
        use std::time::{Duration, Instant};

        let plane = reference_plane(256, 256, 0xf00d);
        let reference = RefPlane::new(&plane, 256, 256);
        let (width, height) = (16, 16);
        let blocks = 8_000;
        let mut timings = Vec::new();
        for level in available_levels() {
            let mut context = McContext::with_level(level);
            let mut dst = vec![0u8; width * height];
            let mut checksum = 0u64;
            // Warm the caches and the branch predictors before timing.
            for _ in 0..64 {
                context.predict_single(
                    reference,
                    8,
                    8,
                    width,
                    height,
                    5,
                    7,
                    InterpFilter::Regular,
                    &mut dst,
                    width,
                );
            }
            // Shared CI runners are noisy, so the best of three trials is
            // reported rather than a single measurement.
            let mut elapsed = Duration::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                for block in 0..blocks {
                    let x = block % 200;
                    let y = (block / 200) % 200;
                    context.predict_single(
                        reference,
                        x,
                        y,
                        width,
                        height,
                        5,
                        7,
                        InterpFilter::Regular,
                        &mut dst,
                        width,
                    );
                    checksum += u64::from(dst[0]);
                }
                elapsed = elapsed.min(start.elapsed());
            }
            assert!(checksum > 0);
            timings.push((level, elapsed));
        }
        let scalar = timings
            .iter()
            .find(|(level, _)| *level == SimdLevel::Scalar)
            .map(|(_, elapsed)| *elapsed)
            .expect("scalar timing");
        for (level, elapsed) in &timings {
            let nanos = elapsed.as_nanos().max(1);
            println!(
                "av1_mc {:>7}: {:>8.1} ns/block, {:>6.2} Mpixel/s, {:.2}x scalar",
                level.name(),
                nanos as f64 / f64::from(blocks),
                (blocks as f64 * (width * height) as f64) / nanos as f64 * 1_000.0,
                scalar.as_nanos() as f64 / nanos as f64,
            );
        }
        // Timing is machine-dependent, so only the ordering is asserted, and
        // only when a vector backend actually exists.
        if timings.len() > 1 {
            let best = timings
                .iter()
                .filter(|(level, _)| *level != SimdLevel::Scalar)
                .map(|(_, elapsed)| *elapsed)
                .min()
                .expect("vector timing");
            assert!(
                best < scalar,
                "no vector backend beat the scalar kernels ({best:?} vs {scalar:?})"
            );
        }
    }
}
