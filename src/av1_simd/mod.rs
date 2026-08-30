//! Runtime-dispatched SIMD kernels for the AV1 transforms and in-loop filters.
//!
//! The AV1 encoder and decoder in this crate are dependency-free pure Rust, so
//! their per-sample inner loops (transform butterflies, deblocking, CDEF, and
//! loop restoration) are where nearly all of the frame time goes. This module
//! adds vectorized implementations of those kernels for SSE4.1 and AVX2 on
//! `x86_64` and NEON on `aarch64`, selected once per process by runtime CPU
//! feature detection, with the existing scalar code kept as the fallback for
//! every other target (including `wasm32`) and for the edge cases the vector
//! kernels deliberately do not cover.
//!
//! # Bit-exactness
//!
//! Every kernel here is a lane-by-lane transliteration of the scalar routine it
//! replaces, so the two produce identical output rather than merely similar
//! output. Where the scalar reference accumulates in `i64`, the vector kernel
//! either proves the `i32` range is sufficient for 8-bit input (the filters) or
//! range-checks its input and defers to the scalar path when the check fails
//! (the transforms; see `transforms::WHT_INPUT_LIMIT` and
//! `transforms::input_limit`). Positions whose taps would need edge
//! clamping stay on the scalar path as well, so the vector kernels never read
//! outside a plane. `tests/av1_simd.rs` asserts equality against the scalar
//! path for every instruction set the host supports.
//!
//! # Selecting an instruction set
//!
//! [`active_isa`] reports what the kernels will use. [`set_active_isa`] forces
//! a specific one (or restores automatic detection with `None`); it exists for
//! the bit-exactness tests and the benchmark, both of which need to run the
//! same input through more than one implementation, and it is safe to call at
//! any time because all implementations agree.

// Targets with no vector implementation (`wasm32` in particular) never
// instantiate the generic kernels and never reach the dispatchers' vector
// arms, so the resulting unused-code warnings are silenced there and only
// there.
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_variables, unreachable_code)
)]

use core::sync::atomic::{AtomicU8, Ordering};

use crate::av1_intra::Tx1d;

pub(crate) mod filters;
pub(crate) mod transforms;
pub(crate) mod vector;

/// An instruction set the AV1 kernels can run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SimdIsa {
    /// The portable scalar reference implementation.
    Scalar,
    /// x86_64 SSE4.1 (128-bit, four 32-bit lanes).
    Sse41,
    /// x86_64 AVX2 (256-bit, eight 32-bit lanes).
    Avx2,
    /// aarch64 NEON (128-bit, four 32-bit lanes).
    Neon,
}

impl SimdIsa {
    /// A short stable name, useful in benchmark and diagnostic output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            SimdIsa::Scalar => "scalar",
            SimdIsa::Sse41 => "sse4.1",
            SimdIsa::Avx2 => "avx2",
            SimdIsa::Neon => "neon",
        }
    }

    fn code(self) -> u8 {
        match self {
            SimdIsa::Scalar => 1,
            SimdIsa::Sse41 => 2,
            SimdIsa::Avx2 => 3,
            SimdIsa::Neon => 4,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(SimdIsa::Scalar),
            2 => Some(SimdIsa::Sse41),
            3 => Some(SimdIsa::Avx2),
            4 => Some(SimdIsa::Neon),
            _ => None,
        }
    }
}

/// Number of samples a single vector operation covers on `isa`, or `0` when
/// `isa` has no vector path and callers should stay scalar.
#[must_use]
pub fn lanes(isa: SimdIsa) -> usize {
    match isa {
        SimdIsa::Scalar => 0,
        SimdIsa::Sse41 | SimdIsa::Neon => 4,
        SimdIsa::Avx2 => 8,
    }
}

/// The best instruction set this CPU supports, ignoring any [`set_active_isa`]
/// override.
#[must_use]
pub fn detected_isa() -> SimdIsa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return SimdIsa::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return SimdIsa::Sse41;
        }
        SimdIsa::Scalar
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory in the aarch64 base architecture, so no runtime
        // probe is needed (and `is_aarch64_feature_detected!` is still
        // unstable).
        SimdIsa::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdIsa::Scalar
    }
}

/// Every instruction set that can be exercised on this host, always including
/// [`SimdIsa::Scalar`]. Used by the bit-exactness tests and the benchmark.
#[must_use]
pub fn available_isas() -> Vec<SimdIsa> {
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_mut)
    )]
    let mut isas = vec![SimdIsa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            isas.push(SimdIsa::Sse41);
        }
        if std::is_x86_feature_detected!("avx2") {
            isas.push(SimdIsa::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        isas.push(SimdIsa::Neon);
    }
    isas
}

static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// The instruction set the kernels will actually use.
#[must_use]
pub fn active_isa() -> SimdIsa {
    match SimdIsa::from_code(OVERRIDE.load(Ordering::Relaxed)) {
        Some(isa) => isa,
        None => detected_isa(),
    }
}

/// Forces every AV1 kernel onto `isa`, or restores automatic detection with
/// `None`.
///
/// Every implementation is bit-exact with every other, so this only changes
/// performance, never output. Passing an instruction set this host does not
/// support is ignored (the kernels fall back to scalar).
pub fn set_active_isa(isa: Option<SimdIsa>) {
    let code = match isa {
        Some(isa) if available_isas().contains(&isa) => isa.code(),
        Some(_) => SimdIsa::Scalar.code(),
        None => 0,
    };
    OVERRIDE.store(code, Ordering::Relaxed);
}

// ---------------------------------------------------------------------
// Per-instruction-set entry points
//
// Each kernel gets one `#[target_feature]` wrapper per instruction set. The
// wrappers are what make the generic kernel bodies legal to run: the intrinsics
// they inline are only valid once the feature is known present, which the
// dispatchers below establish through `active_isa`.
//
// The 4- and 8-point transforms have no useful 256-bit shape (a whole 8x8
// coefficient block is four AVX2 registers), so AVX2 hosts run them through the
// SSE4.1 path and spend the wider registers on the pixel filters instead.
// ---------------------------------------------------------------------

macro_rules! simd_entry_points {
    (
        $(#[$meta:meta])*
        fn [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)?
            = $module:ident::$kernel:ident, avx2 = $avx_vector:ident;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        $(#[$meta])*
        unsafe fn $sse($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::Sse4>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        $(#[$meta])*
        unsafe fn $avx($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::$avx_vector>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        $(#[$meta])*
        unsafe fn $neon($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::Neon>($($arg),*) }
        }
    };
}

simd_entry_points! {
    fn [iwht4x4_sse41, iwht4x4_avx2, iwht4x4_neon](quant: &[i32; 16]) -> [i32; 16]
        = transforms::iwht4x4, avx2 = Sse4;
}
simd_entry_points! {
    fn [fwht4x4_sse41, fwht4x4_avx2, fwht4x4_neon](residual: &[i32; 16]) -> [i32; 16]
        = transforms::fwht4x4, avx2 = Sse4;
}
simd_entry_points! {
    #[allow(clippy::fn_params_excessive_bools)]
    fn [tx4_sse41, tx4_avx2, tx4_neon](
        dequantized: &[i32], column: Tx1d, row: Tx1d,
        lr_flip: bool, ud_flip: bool, out: &mut [i16]
    ) -> bool = transforms::inverse_transform4, avx2 = Sse4;
}
simd_entry_points! {
    #[allow(clippy::fn_params_excessive_bools)]
    fn [tx8_sse41, tx8_avx2, tx8_neon](
        dequantized: &[i32], column: Tx1d, row: Tx1d,
        lr_flip: bool, ud_flip: bool, out: &mut [i16]
    ) -> bool = transforms::inverse_transform8, avx2 = Sse4;
}
simd_entry_points! {
    #[allow(clippy::fn_params_excessive_bools)]
    fn [tx16_sse41, tx16_avx2, tx16_neon](
        dequantized: &[i32], column: Tx1d, row: Tx1d,
        lr_flip: bool, ud_flip: bool, out: &mut [i16]
    ) -> bool = transforms::inverse_transform16, avx2 = Sse4;
}
simd_entry_points! {
    #[allow(clippy::fn_params_excessive_bools)]
    fn [tx32_sse41, tx32_avx2, tx32_neon](
        dequantized: &[i32], column: Tx1d, row: Tx1d,
        lr_flip: bool, ud_flip: bool, out: &mut [i16]
    ) -> bool = transforms::inverse_transform32, avx2 = Sse4;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [deblock_h_sse41, deblock_h_avx2, deblock_h_neon](
        data: &mut [u8], geom: filters::Geometry, x0: usize, y: usize, count: usize,
        limit: i32, blimit: i32, thresh: i32, sizes: &[i32]
    ) = filters::deblock_edge_horizontal, avx2 = Avx2;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [deblock_v_sse41, deblock_v_avx2, deblock_v_neon](
        data: &mut [u8], geom: filters::Geometry, x: usize, y0: usize, count: usize,
        limit: i32, blimit: i32, thresh: i32, sizes: &[i32]
    ) = filters::deblock_edge_vertical, avx2 = Avx2;
}
simd_entry_points! {
    fn [cdef_stats_sse41, cdef_stats_avx2, cdef_stats_neon](
        data: &[u8], geom: filters::Geometry, x0: usize, y0: usize, dr: i32, dc: i32
    ) -> (i32, i32) = filters::cdef_direction_stats, avx2 = Avx2;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [cdef_row_sse41, cdef_row_avx2, cdef_row_neon](
        data: &[u8], geom: filters::Geometry, x0: usize, y: usize, count: usize,
        primary: &[filters::CdefTap], primary_strength: i32, primary_damping_adj: i32,
        secondary: &[filters::CdefTap], secondary_strength: i32, secondary_damping_adj: i32,
        total_weight: i32, dst: &mut [u8]
    ) = filters::cdef_filter_row, avx2 = Avx2;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [wiener_h_sse41, wiener_h_avx2, wiener_h_neon](
        data: &[u8], geom: filters::Geometry, x0: usize, y: usize, count: usize,
        taps: [i32; 3], center_tap: i32, out: &mut [i32]
    ) = filters::wiener_horizontal_row, avx2 = Avx2;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [wiener_v_sse41, wiener_v_avx2, wiener_v_neon](
        intermediate: &[i32], width: usize, height: usize, row: usize, column: usize,
        count: usize, taps: [i32; 3], center_tap: i32, dst: &mut [u8]
    ) = filters::wiener_vertical_row, avx2 = Avx2;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [box_stats_sse41, box_stats_avx2, box_stats_neon](
        data: &[u8], geom: filters::Geometry, x0: usize, y: usize, count: usize,
        radius: usize, sums: &mut [i32], sums_sq: &mut [i32]
    ) = filters::box_stats_row, avx2 = Avx2;
}

/// Expands to a `match` over `$isa` that calls the matching wrapper, evaluating
/// to `$fallback` on scalar or on an instruction set this build cannot reach.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?), $fallback:expr) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => unsafe { $sse($($arg),*) },
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => unsafe { $avx($($arg),*) },
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => unsafe { $neon($($arg),*) },
            _ => $fallback,
        }
    };
}

// ---------------------------------------------------------------------
// Safe dispatchers used by the AV1 transform and filter modules
// ---------------------------------------------------------------------

/// Vectorized [`crate::av1_encoder`] inverse WHT, or `None` when the caller
/// should use the scalar path.
pub(crate) fn iwht4x4(isa: SimdIsa, quant: &[i32; 16]) -> Option<[i32; 16]> {
    if !transforms::within_limit(quant, transforms::WHT_INPUT_LIMIT) {
        return None;
    }
    let residual: [i32; 16] = dispatch!(
        isa,
        [iwht4x4_sse41, iwht4x4_avx2, iwht4x4_neon](quant),
        return None
    );
    Some(residual)
}

/// Vectorized [`crate::av1_encoder`] forward WHT, or `None` when the caller
/// should use the scalar path.
pub(crate) fn fwht4x4(isa: SimdIsa, residual: &[i32; 16]) -> Option<[i32; 16]> {
    if !transforms::within_limit(residual, transforms::WHT_INPUT_LIMIT) {
        return None;
    }
    let coefficients: [i32; 16] = dispatch!(
        isa,
        [fwht4x4_sse41, fwht4x4_avx2, fwht4x4_neon](residual),
        return None
    );
    Some(coefficients)
}

/// Vectorized non-lossless inverse transform for one `size x size` block.
///
/// `column` and `row` are the vertical and horizontal 1-D kernels and
/// `lr_flip`/`ud_flip` the flipped-ADST output reversals, as reported by
/// [`crate::av1_intra::Av1TxType::kernels`]. Returns `false` (leaving `out`
/// untouched) when the caller should use the scalar path: an unsupported
/// size, an instruction set this build cannot reach, or coefficients large
/// enough that a 32-bit lane could overflow.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(crate) fn inverse_transform(
    isa: SimdIsa,
    dequantized: &[i32],
    size: usize,
    column: Tx1d,
    row: Tx1d,
    lr_flip: bool,
    ud_flip: bool,
    out: &mut [i16],
) -> bool {
    if !transforms::within_limit(dequantized, transforms::input_limit(size)) {
        return false;
    }
    let arguments = (dequantized, column, row, lr_flip, ud_flip, out);
    match size {
        4 => dispatch!(
            isa,
            [tx4_sse41, tx4_avx2, tx4_neon](
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5
            ),
            false
        ),
        8 => dispatch!(
            isa,
            [tx8_sse41, tx8_avx2, tx8_neon](
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5
            ),
            false
        ),
        16 => dispatch!(
            isa,
            [tx16_sse41, tx16_avx2, tx16_neon](
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5
            ),
            false
        ),
        32 => dispatch!(
            isa,
            [tx32_sse41, tx32_avx2, tx32_neon](
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5
            ),
            false
        ),
        _ => false,
    }
}

/// Filters `count` consecutive positions of the horizontal edge above row `y`,
/// including positions whose filter window leaves the plane; `sizes` gives each
/// position's §7.14.5 filter length, or is empty when every edge is narrow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_edge_horizontal(
    isa: SimdIsa,
    data: &mut [u8],
    geom: filters::Geometry,
    x0: usize,
    y: usize,
    count: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    sizes: &[i32],
) {
    dispatch!(
        isa,
        [deblock_h_sse41, deblock_h_avx2, deblock_h_neon](
            data, geom, x0, y, count, limit, blimit, thresh, sizes
        ),
        ()
    )
}

/// Filters `count` consecutive positions of the vertical edge left of column
/// `x`, including positions whose filter window leaves the plane; `sizes` gives
/// each position's §7.14.5 filter length, or is empty when every edge is
/// narrow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_edge_vertical(
    isa: SimdIsa,
    data: &mut [u8],
    geom: filters::Geometry,
    x: usize,
    y0: usize,
    count: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    sizes: &[i32],
) {
    dispatch!(
        isa,
        [deblock_v_sse41, deblock_v_avx2, deblock_v_neon](
            data, geom, x, y0, count, limit, blimit, thresh, sizes
        ),
        ()
    )
}

/// CDEF direction-search statistics for one 8x8 block along `(dr, dc)`, or
/// `None` when the caller should use the scalar path.
pub(crate) fn cdef_direction_stats(
    isa: SimdIsa,
    data: &[u8],
    geom: filters::Geometry,
    x0: usize,
    y0: usize,
    dr: i32,
    dc: i32,
) -> Option<(i32, i32)> {
    let stats: (i32, i32) = dispatch!(
        isa,
        [cdef_stats_sse41, cdef_stats_avx2, cdef_stats_neon](data, geom, x0, y0, dr, dc),
        return None
    );
    Some(stats)
}

/// CDEF-filters `count` consecutive samples of row `y` starting at column `x0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_row(
    isa: SimdIsa,
    data: &[u8],
    geom: filters::Geometry,
    x0: usize,
    y: usize,
    count: usize,
    primary: &[filters::CdefTap],
    primary_strength: i32,
    secondary: &[filters::CdefTap],
    secondary_strength: i32,
    damping: i32,
    total_weight: i32,
    dst: &mut [u8],
) {
    let primary_adj = filters::constrain_damping_adjustment(primary_strength, damping);
    let secondary_adj = filters::constrain_damping_adjustment(secondary_strength, damping);
    dispatch!(
        isa,
        [cdef_row_sse41, cdef_row_avx2, cdef_row_neon](
            data,
            geom,
            x0,
            y,
            count,
            primary,
            primary_strength,
            primary_adj,
            secondary,
            secondary_strength,
            secondary_adj,
            total_weight,
            dst
        ),
        ()
    )
}

/// Wiener horizontal pass for `count` consecutive samples of row `y`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wiener_horizontal_row(
    isa: SimdIsa,
    data: &[u8],
    geom: filters::Geometry,
    x0: usize,
    y: usize,
    count: usize,
    taps: [i32; 3],
    center_tap: i32,
    out: &mut [i32],
) {
    dispatch!(
        isa,
        [wiener_h_sse41, wiener_h_avx2, wiener_h_neon](
            data, geom, x0, y, count, taps, center_tap, out
        ),
        ()
    )
}

/// Wiener vertical pass for `count` consecutive columns of `row`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wiener_vertical_row(
    isa: SimdIsa,
    intermediate: &[i32],
    width: usize,
    height: usize,
    row: usize,
    column: usize,
    count: usize,
    taps: [i32; 3],
    center_tap: i32,
    dst: &mut [u8],
) {
    dispatch!(
        isa,
        [wiener_v_sse41, wiener_v_avx2, wiener_v_neon](
            intermediate,
            width,
            height,
            row,
            column,
            count,
            taps,
            center_tap,
            dst
        ),
        ()
    )
}

/// Self-guided box statistics for `count` consecutive samples of row `y`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_stats_row(
    isa: SimdIsa,
    data: &[u8],
    geom: filters::Geometry,
    x0: usize,
    y: usize,
    count: usize,
    radius: usize,
    sums: &mut [i32],
    sums_sq: &mut [i32],
) {
    dispatch!(
        isa,
        [box_stats_sse41, box_stats_avx2, box_stats_neon](
            data, geom, x0, y, count, radius, sums, sums_sq
        ),
        ()
    )
}

/// Widest lane count any implementation uses; the size scratch buffers need.
pub(crate) const MAX_LANES: usize = vector::MAX_LANES;

#[cfg(test)]
mod tests;
