//! Bit-exactness coverage: every vectorized kernel must produce output
//! identical to the scalar reference, for every instruction set the host
//! supports.
//!
//! The end-to-end filter tests drive the public filter entry points with
//! [`set_active_isa`] pinned to each instruction set in turn, which exercises
//! the dispatch conditions in `av1_filters` (run eligibility, edge handling,
//! and the scalar tail) as well as the kernels themselves. They share
//! [`ISA_LOCK`] so a pinned instruction set is not swapped out underneath a
//! test by a concurrently running one - and, since the override is now
//! crate-wide, by the HEVC tests that pin the scalar path as well.

use std::sync::MutexGuard;

use super::forward_transform as forward_transform_simd;
use super::inverse_transform as inverse_transform_simd;
use super::*;
use crate::av1_encoder::transform::forward_transform;
use crate::av1_encoder::wht::{fwht4x4_scalar, iwht4x4_scalar};
use crate::av1_filters::{
    CdefStrength, FilterFrame, FilterPlane, LoopFilterParams, RestorationUnit,
    apply_restoration_unit, cdef_frame, deblock_frame,
};
use crate::av1_intra::{Av1TxType, Tx1d, inverse_transform};
use crate::{Limits, TxSizeGrid};

/// Delegates to the crate-wide override lock: pinning an instruction set now
/// changes every codec's kernels, not just this module's, so the HEVC tests
/// that pin the scalar path have to exclude these too.
fn lock_isa() -> MutexGuard<'static, ()> {
    crate::simd::test_lock()
}

/// Small deterministic LCG, matching the style used elsewhere in the crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }

    fn in_range(&mut self, span: i32) -> i32 {
        (self.next() >> 33) as i32 % (2 * span + 1) - span
    }
}

fn plane(width: usize, height: usize, seed: u64) -> FilterPlane {
    let mut rng = Lcg(seed);
    let data = (0..width * height).map(|_| rng.byte()).collect();
    FilterPlane::from_samples(width, height, data, &Limits::default()).unwrap()
}

/// A plane of near-flat blocks separated by small steps: the wide deblocking
/// filters are gated on §7.14.6.1's flatness check, so uniformly random content
/// never reaches them. Each 16x16 block is constant apart from a one-level
/// ripple, which is inside `FLAT_THRESH`, and neighboring blocks differ by a
/// small step so the boundary masks still have something to smooth.
fn flat_blocks_plane(width: usize, height: usize, seed: u64) -> FilterPlane {
    let mut rng = Lcg(seed);
    let data = (0..width * height)
        .map(|index| {
            let (x, y) = (index % width, index / width);
            let block = ((x / 16 + y / 16) % 5) as i32;
            let ripple = (rng.byte() % 2) as i32;
            (100 + block * 6 + ripple) as u8
        })
        .collect();
    FilterPlane::from_samples(width, height, data, &Limits::default()).unwrap()
}

/// Runs `body` once per available instruction set, returning the results in
/// the same order as [`available_isas`].
fn for_each_isa<T>(mut body: impl FnMut(SimdIsa) -> T) -> Vec<(SimdIsa, T)> {
    let _guard = lock_isa();
    let results = available_isas()
        .into_iter()
        .map(|isa| {
            set_active_isa(Some(isa));
            assert_eq!(active_isa(), isa, "override should pin the instruction set");
            (isa, body(isa))
        })
        .collect();
    set_active_isa(None);
    results
}

fn assert_all_match<T: PartialEq + std::fmt::Debug>(results: &[(SimdIsa, T)], what: &str) {
    let (reference_isa, reference) = &results[0];
    assert_eq!(*reference_isa, SimdIsa::Scalar);
    for (isa, value) in &results[1..] {
        assert_eq!(
            value,
            reference,
            "{what} differs between {} and {}",
            isa.name(),
            reference_isa.name()
        );
    }
}

// ---------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------

/// Whether `fwht4x4` is expected to answer with a kernel result. The forward
/// WHT is dispatched to the scalar reference on x86_64 (see `super::fwht4x4`),
/// so there `None` is the documented answer for an in-range block rather than
/// a fallback. The inverse still has a kernel everywhere.
const FORWARD_WHT_HAS_KERNEL: bool = !cfg!(target_arch = "x86_64");

#[test]
fn walsh_hadamard_kernels_match_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0120_0000_0001);
    for isa in available_isas() {
        if lanes(isa) == 0 {
            continue;
        }
        for _ in 0..2_000 {
            let mut residual = [0i32; 16];
            for value in &mut residual {
                *value = rng.in_range(255);
            }
            let scalar_coefficients = fwht4x4_scalar(&residual);
            let kernel = fwht4x4(isa, &residual);
            assert_eq!(
                kernel.is_some(),
                FORWARD_WHT_HAS_KERNEL,
                "{}: unexpected forward WHT dispatch",
                isa.name()
            );
            let coefficients = kernel.unwrap_or(scalar_coefficients);
            assert_eq!(coefficients, scalar_coefficients, "{}", isa.name());
            let reconstructed = iwht4x4(isa, &coefficients).expect("in-range block");
            assert_eq!(
                reconstructed,
                iwht4x4_scalar(&coefficients),
                "{}",
                isa.name()
            );
            assert_eq!(reconstructed, residual, "forward/inverse must round-trip");
        }
    }
}

#[test]
fn out_of_range_walsh_hadamard_blocks_fall_back_to_scalar() {
    for isa in available_isas() {
        let mut block = [1i32; 16];
        block[7] = transforms::WHT_INPUT_LIMIT + 1;
        assert!(iwht4x4(isa, &block).is_none());
        assert!(fwht4x4(isa, &block).is_none());
    }
}

/// Every transform type this crate implements, paired with the sizes it is
/// defined at. ADST has no 32- or 64-point kernel in AV1.
const TX_TYPES: [(Av1TxType, &[usize]); 16] = [
    (Av1TxType::Idtx, &[4, 8, 16, 32, 64]),
    (Av1TxType::DctDct, &[4, 8, 16, 32, 64]),
    (Av1TxType::AdstDct, &[4, 8, 16]),
    (Av1TxType::DctAdst, &[4, 8, 16]),
    (Av1TxType::AdstAdst, &[4, 8, 16]),
    (Av1TxType::FlipadstDct, &[4, 8, 16]),
    (Av1TxType::DctFlipadst, &[4, 8, 16]),
    (Av1TxType::FlipadstFlipadst, &[4, 8, 16]),
    (Av1TxType::AdstFlipadst, &[4, 8, 16]),
    (Av1TxType::FlipadstAdst, &[4, 8, 16]),
    (Av1TxType::VDct, &[4, 8, 16, 32, 64]),
    (Av1TxType::HDct, &[4, 8, 16, 32, 64]),
    (Av1TxType::VAdst, &[4, 8, 16]),
    (Av1TxType::HAdst, &[4, 8, 16]),
    (Av1TxType::VFlipadst, &[4, 8, 16]),
    (Av1TxType::HFlipadst, &[4, 8, 16]),
];

#[test]
fn inverse_transforms_match_the_scalar_reference_at_every_size_and_type() {
    let mut rng = Lcg(0x5eed_0120_0000_0002);
    for (tx_type, sizes) in TX_TYPES {
        for &size in sizes {
            for _ in 0..40 {
                let coefficients: Vec<i32> = (0..size * size).map(|_| rng.in_range(600)).collect();
                let results =
                    for_each_isa(|_| inverse_transform(&coefficients, size, tx_type, 20, 14));
                assert_all_match(&results, &format!("{tx_type:?} {size}x{size} output"));
            }
        }
    }
}

/// The widened accumulators are only worth anything if they stay bit-exact
/// right up to the documented bound, so this drives every size and type with
/// coefficients placed exactly at it, in the sign patterns that maximize each
/// butterfly stage.
#[test]
fn inverse_transforms_stay_bit_exact_at_the_documented_input_limit() {
    let mut rng = Lcg(0x5eed_0120_0000_0003);
    for (tx_type, sizes) in TX_TYPES {
        for &size in sizes {
            let limit = transforms::input_limit(size);
            // The dequantizer multiplies by `ac_quant`, so feed coefficients
            // that land exactly on the limit after scaling.
            let extreme = limit / 4;
            for pattern in 0..6 {
                let coefficients: Vec<i32> = (0..size * size)
                    .map(|index| {
                        let sign = match pattern {
                            0 => 1,
                            1 => -1,
                            2 => {
                                if index % 2 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            3 => {
                                if (index / size) % 2 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            4 => {
                                if index % size < size / 2 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            _ => {
                                if rng.next() & 1 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                        };
                        sign * extreme
                    })
                    .collect();
                let results =
                    for_each_isa(|_| inverse_transform(&coefficients, size, tx_type, 4, 4));
                assert_all_match(
                    &results,
                    &format!("{tx_type:?} {size}x{size} at the input limit"),
                );
            }
        }
    }
}

#[test]
fn out_of_range_transform_blocks_fall_back_to_scalar() {
    for size in [4usize, 8, 16, 32, 64] {
        let over = transforms::input_limit(size) as i64 + 1;
        let coefficients: Vec<i32> = (0..size * size)
            .map(|index| if index == 3 { over as i32 } else { 1 })
            .collect();
        let results = for_each_isa(|isa| {
            let mut out = vec![0i16; size * size];
            let vectorized = inverse_transform_simd(
                isa,
                &coefficients,
                size,
                Tx1d::Dct,
                Tx1d::Dct,
                false,
                false,
                &mut out,
            );
            assert!(!vectorized, "the range guard must reject this block");
            inverse_transform(&coefficients, size, Av1TxType::DctDct, 1, 1)
        });
        assert_all_match(&results, "out-of-range inverse transform output");
    }
}

/// A bit-exactness comparison passes vacuously if every backend quietly took
/// the scalar path, so pin down that each size really is dispatched to a
/// vector kernel for ordinary coefficient magnitudes.
#[test]
fn every_transform_size_reaches_a_vector_kernel() {
    let mut rng = Lcg(0x5eed_0120_0000_0005);
    for (tx_type, sizes) in TX_TYPES {
        let (column, row, lr_flip, ud_flip) = tx_type.kernels();
        for &size in sizes {
            let coefficients: Vec<i32> = (0..size * size).map(|_| rng.in_range(600)).collect();
            for (isa, vectorized) in for_each_isa(|isa| {
                let mut out = vec![0i16; size * size];
                inverse_transform_simd(
                    isa,
                    &coefficients,
                    size,
                    column,
                    row,
                    lr_flip,
                    ud_flip,
                    &mut out,
                )
            }) {
                assert_eq!(
                    vectorized,
                    isa != SimdIsa::Scalar,
                    "{tx_type:?} {size}x{size} on {}",
                    isa.name()
                );
            }
        }
    }
}

/// The fixed-point butterflies must actually compute the transforms they
/// claim to: `idctN` is the AV1/VP9-lineage DCT-III, `iadst4` is the DST-VII
/// the `sinpi` constants encode, and `iadst8`/`iadst16` are DST-IV. Checking
/// against a direct double-precision evaluation catches a mistranscribed
/// butterfly or constant that a scalar-versus-SIMD comparison alone cannot.
#[test]
fn scalar_kernels_match_the_mathematical_transforms() {
    use std::f64::consts::{PI, SQRT_2};

    for size in [4usize, 8, 16, 32, 64] {
        for basis in 0..size {
            let mut coefficients = vec![0i32; size * size];
            // A single row coefficient, so the column pass only scales the DC
            // basis and the row pass is what is being measured.
            coefficients[basis] = 1;
            let residual =
                inverse_transform(&coefficients, size, Av1TxType::DctDct, 1 << 12, 1 << 12);
            let column_gain = 1.0 / SQRT_2;
            let shift = f64::from(1 << crate::av1_intra::transform_shift(size));
            for (n, &sample) in residual.iter().take(size).enumerate() {
                let expected = f64::from(1 << 12)
                    * if basis == 0 {
                        1.0 / SQRT_2
                    } else {
                        (PI * (2.0 * n as f64 + 1.0) * basis as f64 / (2.0 * size as f64)).cos()
                    }
                    * column_gain
                    / shift;
                let got = f64::from(sample);
                assert!(
                    (got - expected).abs() <= 2.0,
                    "dct{size} basis {basis} position {n}: {got} vs {expected}"
                );
            }
        }
    }

    for size in [4usize, 8, 16] {
        for basis in 0..size {
            let mut coefficients = vec![0i32; size * size];
            coefficients[basis] = 1;
            let residual =
                inverse_transform(&coefficients, size, Av1TxType::DctAdst, 1 << 12, 1 << 12);
            let shift = f64::from(1 << crate::av1_intra::transform_shift(size));
            for (n, &sample) in residual.iter().take(size).enumerate() {
                let sine = if size == 4 {
                    // DST-VII, scaled by `2 * sqrt(2) / 3` by the `sinpi`
                    // constants.
                    2.0 * SQRT_2 / 3.0
                        * (PI * (n as f64 + 1.0) * (2.0 * basis as f64 + 1.0) / 9.0).sin()
                } else {
                    // DST-IV.
                    (PI * (2.0 * n as f64 + 1.0) * (2.0 * basis as f64 + 1.0) / (4.0 * size as f64))
                        .sin()
                };
                let expected = f64::from(1 << 12) * sine / SQRT_2 / shift;
                let got = f64::from(sample);
                assert!(
                    (got - expected).abs() <= 2.0,
                    "adst{size} basis {basis} position {n}: {got} vs {expected}"
                );
            }
        }
    }
}

/// The flipped-ADST types are the plain ADST block reversed along one or both
/// axes, so they must agree with an explicit reversal of the unflipped result.
#[test]
fn flipped_adst_reverses_the_unflipped_block() {
    let mut rng = Lcg(0x5eed_0120_0000_0004);
    for size in [4usize, 8, 16] {
        let coefficients: Vec<i32> = (0..size * size).map(|_| rng.in_range(600)).collect();
        let plain = inverse_transform(&coefficients, size, Av1TxType::AdstAdst, 20, 14);
        for (tx_type, lr, ud) in [
            (Av1TxType::AdstFlipadst, true, false),
            (Av1TxType::FlipadstAdst, false, true),
            (Av1TxType::FlipadstFlipadst, true, true),
        ] {
            let flipped = inverse_transform(&coefficients, size, tx_type, 20, 14);
            for row in 0..size {
                for column in 0..size {
                    let source_row = if ud { size - 1 - row } else { row };
                    let source_column = if lr { size - 1 - column } else { column };
                    assert_eq!(
                        flipped[row * size + column],
                        plain[source_row * size + source_column],
                        "{tx_type:?} {size}x{size} at ({row}, {column})"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// In-loop filters
// ---------------------------------------------------------------------

fn deblock_params(level: u8, sharpness: u8) -> LoopFilterParams {
    LoopFilterParams {
        y_vertical_level: level,
        y_horizontal_level: level,
        u_level: level,
        v_level: level,
        sharpness,
    }
}

#[test]
fn deblocking_matches_the_scalar_reference() {
    for (width, height) in [(64usize, 48usize), (37, 29), (9, 7)] {
        for (level, sharpness) in [(16u8, 0u8), (32, 3), (63, 7)] {
            let results = for_each_isa(|_| {
                let mut frame =
                    FilterFrame::new_monochrome(plane(width, height, 0x1234 + level as u64));
                deblock_frame(&mut frame, &deblock_params(level, sharpness), None).unwrap();
                frame.y.data
            });
            assert_all_match(&results, "deblocked plane");
        }
    }
}

#[test]
fn deblocking_with_mixed_transform_sizes_matches_the_scalar_reference() {
    // A grid that mixes 4x4, 16x16, and 32x32 blocks forces runs that are only
    // partly eligible for the narrow-filter vector path, exercising both the
    // vector path and the scalar tail that handles the wide filters.
    let (width, height) = (64usize, 64usize);
    let results = for_each_isa(|_| {
        let mut grid = TxSizeGrid::new(width, height);
        grid.set_block(0, 0, 32, 32);
        grid.set_block(32, 0, 16, 16);
        grid.set_block(0, 32, 4, 4);
        grid.set_block(32, 32, 8, 8);
        let mut frame = FilterFrame::new_monochrome(plane(width, height, 0x99));
        deblock_frame(&mut frame, &deblock_params(40, 2), Some(&grid)).unwrap();
        frame.y.data
    });
    assert_all_match(&results, "deblocked plane with transform sizes");
}

#[test]
fn cdef_matches_the_scalar_reference() {
    for (width, height) in [(64usize, 64usize), (33, 17)] {
        for (primary, secondary, damping) in [(4u8, 2u8, 3u8), (15, 0, 6), (0, 4, 5)] {
            let results = for_each_isa(|_| {
                let strength = CdefStrength {
                    y_primary: primary,
                    y_secondary: secondary,
                    uv_primary: primary,
                    uv_secondary: secondary,
                    damping,
                };
                let mut frame = FilterFrame::new_monochrome(plane(width, height, 0xabcd));
                cdef_frame(&mut frame, &strength, &Limits::default()).unwrap();
                frame.y.data
            });
            assert_all_match(&results, "CDEF filtered plane");
        }
    }
}

#[test]
fn wiener_restoration_matches_the_scalar_reference() {
    for (width, height) in [(64usize, 40usize), (11, 9)] {
        let results = for_each_isa(|_| {
            let mut target = plane(width, height, 0x2468);
            let unit = RestorationUnit::Wiener {
                horizontal: [3, -7, 15],
                vertical: [-2, 5, 11],
            };
            apply_restoration_unit(&mut target, &unit, 0, 0, width, height).unwrap();
            target.data
        });
        assert_all_match(&results, "Wiener restored plane");
    }
}

#[test]
fn self_guided_restoration_matches_the_scalar_reference() {
    for radius in [[1u8, 2u8], [3, 0], [2, 3]] {
        let results = for_each_isa(|_| {
            let mut target = plane(48, 36, 0x1357);
            let unit = RestorationUnit::SelfGuided {
                radius,
                eps: [12, 30],
                weight: [40, 24],
            };
            apply_restoration_unit(&mut target, &unit, 2, 3, 46, 33).unwrap();
            target.data
        });
        assert_all_match(&results, "self-guided restored plane");
    }
}

/// A transform-size grid whose block sizes change at 64-sample (superblock and
/// tile) boundaries, so edges select all three filter lengths and a single
/// vector run can straddle two different ones.
fn mixed_tx_grid(width: usize, height: usize) -> TxSizeGrid {
    let mut grid = TxSizeGrid::new(width, height);
    for y in (0..height).step_by(64) {
        for x in (0..width).step_by(64) {
            let size = match ((x / 64) + (y / 64)) % 4 {
                0 => 32,
                1 => 16,
                2 => 8,
                _ => 4,
            };
            let mut by = y;
            while by < (y + 64).min(height) {
                let mut bx = x;
                while bx < (x + 64).min(width) {
                    grid.set_block(bx, by, size, size);
                    bx += size;
                }
                by += size;
            }
        }
    }
    grid
}

#[test]
fn wide_deblocking_filters_match_the_scalar_reference() {
    for (width, height) in [(96usize, 96usize), (67, 71), (33, 18)] {
        for (level, sharpness) in [(16u8, 0u8), (40, 2), (63, 7)] {
            let grid = mixed_tx_grid(width, height);
            let results = for_each_isa(|_| {
                let mut frame =
                    FilterFrame::new_monochrome(flat_blocks_plane(width, height, 0x5eed));
                deblock_frame(&mut frame, &deblock_params(level, sharpness), Some(&grid)).unwrap();
                frame.y.data
            });
            assert_all_match(&results, "deblocked plane with wide filters");
        }
    }
}

#[test]
fn wide_deblocking_filters_actually_run_on_flat_content() {
    // Guards the coverage above: if the flatness gate never opened, every
    // result would equal the narrow-only run and the test would be vacuous.
    let (width, height) = (96usize, 96usize);
    let grid = mixed_tx_grid(width, height);
    let mut wide = FilterFrame::new_monochrome(flat_blocks_plane(width, height, 0x5eed));
    deblock_frame(&mut wide, &deblock_params(40, 2), Some(&grid)).unwrap();
    let mut narrow = FilterFrame::new_monochrome(flat_blocks_plane(width, height, 0x5eed));
    deblock_frame(&mut narrow, &deblock_params(40, 2), None).unwrap();
    assert_ne!(
        wide.y.data, narrow.y.data,
        "the 8-tap/14-tap filters should change samples the narrow filter does not"
    );

    // The two grids differ only in whether an edge selects the 14-tap or the
    // 8-tap filter, so a difference here means the 14-tap path ran too.
    let uniform = |size: usize| {
        let mut grid = TxSizeGrid::new(width, height);
        for y in (0..height).step_by(size) {
            for x in (0..width).step_by(size) {
                grid.set_block(x, y, size, size);
            }
        }
        let mut frame = FilterFrame::new_monochrome(flat_blocks_plane(width, height, 0x5eed));
        deblock_frame(&mut frame, &deblock_params(40, 2), Some(&grid)).unwrap();
        frame.y.data
    };
    assert_ne!(
        uniform(16),
        uniform(8),
        "the 14-tap filter should reach samples the 8-tap filter does not"
    );
    // 32x32 and 16x16 transforms both clamp to §7.14.5's `filterSize == 16`,
    // so they must deblock identically.
    assert_eq!(
        uniform(32),
        uniform(16),
        "filterSize clamps at 16, so 32x32 and 16x16 transforms filter alike"
    );
}

/// Verifies wide-filter output against spec-derived values rather than
/// against this crate's own scalar path, on every instruction set.
///
/// The plane is a single step edge: columns `0..16` hold 100 and columns
/// `16..20` hold 110, with a frame-wide 32x32 transform grid so §7.14.5
/// selects the 14-tap filter. Height 4 leaves no horizontal edge, and the
/// vertical edges at x = 4, 8 and 12 see an all-100 window, which any filter
/// whose weights sum to its rounding shift leaves unchanged. Only the last
/// edge, x = 16, actually filters, so no cascade obscures its output.
///
/// The expected samples come from §7.14.6.4 directly: with p6..p0 = 100 and
/// q0..q6 = 110 (the taps past the right border replicate column 19), output
/// `k` is `Round2(100 * (16 - w) + 110 * w, 4)` where `w` is the row's q-side
/// weight sum, i.e. `(1608 + 10 * w) >> 4`. The q-side weight sums read off
/// the spec's tap lists are 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14 and 15.
#[test]
fn wide_deblocking_output_matches_spec_derived_vectors() {
    const WIDTH: usize = 20;
    const HEIGHT: usize = 4;
    // Written at columns 10..=21; the last two land outside the plane.
    const Q_WEIGHT_SUMS: [i32; 12] = [1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15];

    let mut expected: Vec<u8> = (0..WIDTH)
        .map(|x| if x < 16 { 100u8 } else { 110u8 })
        .collect();
    for (offset, weight) in Q_WEIGHT_SUMS.into_iter().enumerate() {
        let column = 10 + offset;
        if column < WIDTH {
            expected[column] = ((1608 + 10 * weight) >> 4) as u8;
        }
    }
    let expected: Vec<u8> = expected.repeat(HEIGHT);

    let mut grid = TxSizeGrid::new(WIDTH, HEIGHT);
    grid.set_block(0, 0, 32, 32);
    let results = for_each_isa(|_| {
        let data: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|index| if index % WIDTH < 16 { 100u8 } else { 110u8 })
            .collect();
        let plane = FilterPlane::from_samples(WIDTH, HEIGHT, data, &Limits::default()).unwrap();
        let mut frame = FilterFrame::new_monochrome(plane);
        deblock_frame(&mut frame, &deblock_params(40, 0), Some(&grid)).unwrap();
        frame.y.data
    });
    for (isa, data) in &results {
        assert_eq!(
            data,
            &expected,
            "{} wide-filter output should match the spec's filter14 constants",
            isa.name()
        );
    }
}

/// A 4:2:0 frame whose luma and chroma planes are both near-flat blocks, so
/// the chroma flatness gate opens and the 6-tap filter actually runs.
fn flat_blocks_frame(width: usize, height: usize, seed: u64) -> FilterFrame {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    FilterFrame::new_yuv(
        flat_blocks_plane(width, height, seed),
        flat_blocks_plane(cw, ch, seed ^ 0x11),
        flat_blocks_plane(cw, ch, seed ^ 0x22),
        true,
        true,
    )
    .unwrap()
}

#[test]
fn chroma_deblocking_matches_the_scalar_reference() {
    // Chroma planes take the 6-tap filter where the subsampled transform
    // sizes select it, so both the vector kernel's 6-tap lane path and the
    // scalar fallback must agree on every instruction set.
    for (width, height) in [(96usize, 96usize), (67, 71), (33, 18), (9, 7)] {
        for (level, sharpness) in [(16u8, 0u8), (40, 2), (63, 7)] {
            let grid = mixed_tx_grid(width, height);
            let results = for_each_isa(|_| {
                let mut frame = flat_blocks_frame(width, height, 0x5eed);
                deblock_frame(&mut frame, &deblock_params(level, sharpness), Some(&grid)).unwrap();
                let u = frame.u.unwrap().data;
                let v = frame.v.unwrap().data;
                (u, v)
            });
            assert_all_match(&results, "deblocked chroma planes");
        }
    }
}

#[test]
fn chroma_wide_deblocking_actually_runs_on_flat_content() {
    // Guards the coverage above: without this the chroma comparison could
    // pass with the 6-tap path never selected.
    let (width, height) = (96usize, 96usize);
    let grid = mixed_tx_grid(width, height);
    let mut wide = flat_blocks_frame(width, height, 0x5eed);
    deblock_frame(&mut wide, &deblock_params(40, 2), Some(&grid)).unwrap();
    let mut narrow = flat_blocks_frame(width, height, 0x5eed);
    deblock_frame(&mut narrow, &deblock_params(40, 2), None).unwrap();
    assert_ne!(
        wide.u.unwrap().data,
        narrow.u.unwrap().data,
        "the 6-tap chroma filter should change samples the narrow filter does not"
    );
}

/// A grid of 16x16 luma transforms, whose subsampled 8x8 chroma transforms make
/// §7.14.5 select the 6-tap filter on every interior chroma edge.
fn luma_16x16_grid(width: usize, height: usize) -> TxSizeGrid {
    let mut grid = TxSizeGrid::new(width, height);
    for y in (0..height).step_by(16) {
        for x in (0..width).step_by(16) {
            grid.set_block(x, y, 16, 16);
        }
    }
    grid
}

/// The 6-tap chunk path loads only `p2..q2`, which is sound exactly because
/// `filter6` and its gates read nothing else. This pins that down from the
/// outside: perturbing every chroma row the window does not cover must leave
/// the four rows the edge writes bit-identical.
///
/// The chroma planes are `24x16` with 8x8 chroma transforms, so the only
/// interior horizontal chroma edge is at `y = 8`; its window is rows `5..=10`
/// and it writes rows `6..=9`. Each row is constant across the plane, so the
/// vertical chroma edges see flat windows and leave the samples alone, and the
/// perturbation stays outside the window on both sides.
#[test]
fn chroma_six_tap_output_depends_only_on_the_six_taps() {
    const CW: usize = 24;
    const CH: usize = 16;
    /// Rows the 6-tap window covers, and the rows the edge writes.
    const WINDOW: [usize; 2] = [5, 10];
    const WRITTEN: [usize; 2] = [6, 10];

    let deblocked = |row_value: &dyn Fn(usize) -> u8| {
        let grid = luma_16x16_grid(2 * CW, 2 * CH);
        let chroma = || {
            let data: Vec<u8> = (0..CW * CH).map(|index| row_value(index / CW)).collect();
            FilterPlane::from_samples(CW, CH, data, &Limits::default()).unwrap()
        };
        let mut frame = FilterFrame::new_yuv(
            plane(2 * CW, 2 * CH, 0x77aa),
            chroma(),
            chroma(),
            true,
            true,
        )
        .unwrap();
        deblock_frame(&mut frame, &deblock_params(40, 0), Some(&grid)).unwrap();
        frame.u.unwrap().data
    };
    let rows = |data: &[u8], range: [usize; 2]| data[range[0] * CW..range[1] * CW].to_vec();

    let base = |row: usize| if row < 8 { 100u8 } else { 110u8 };
    let perturbed = |row: usize| {
        if (WINDOW[0]..=WINDOW[1]).contains(&row) {
            base(row)
        } else {
            base(row) ^ 0x5a
        }
    };

    let results = for_each_isa(|_| (deblocked(&base), deblocked(&perturbed)));
    for (isa, (plain, poisoned)) in &results {
        assert_eq!(
            rows(plain, WRITTEN),
            rows(poisoned, WRITTEN),
            "{} 6-tap chroma output should not depend on samples outside p2..q2",
            isa.name()
        );
        // Neither half of the comparison may be vacuous: the perturbation has
        // to reach the plane, and the 6-tap edge has to have filtered.
        assert_ne!(
            rows(plain, [0, 5]),
            rows(poisoned, [0, 5]),
            "{} the poisoned rows should differ, or nothing was perturbed",
            isa.name()
        );
        let unfiltered: Vec<u8> = (WRITTEN[0]..WRITTEN[1])
            .flat_map(|row| vec![base(row); CW])
            .collect();
        assert_ne!(
            rows(plain, WRITTEN),
            unfiltered,
            "{} the 6-tap chroma edge should have filtered",
            isa.name()
        );
    }
}

/// Chroma planes whose 6-tap window runs off the plane: with 8x8 chroma
/// transforms the horizontal edge at `y = 8` reads up to row 10, so a chroma
/// height of 10 or 11 puts `q2` outside the plane and the reduced-window path
/// must replicate the nearest in-plane sample exactly as the scalar path does.
#[test]
fn chroma_six_tap_at_plane_borders_matches_the_scalar_reference() {
    for (width, height) in [(40usize, 20usize), (36, 22), (34, 19), (22, 21)] {
        let grid = luma_16x16_grid(width, height);
        let results = for_each_isa(|_| {
            let mut frame = flat_blocks_frame(width, height, 0x60_1de5);
            deblock_frame(&mut frame, &deblock_params(40, 0), Some(&grid)).unwrap();
            (frame.u.unwrap().data, frame.v.unwrap().data)
        });
        assert_all_match(&results, "deblocked chroma planes at plane borders");
    }
}

/// Verifies 6-tap chroma output against spec-derived values rather than this
/// crate's own scalar path, on every instruction set.
///
/// The 4:2:0 chroma planes are a single step edge: columns `0..16` hold 100
/// and `16..20` hold 110, with 16x16 luma transforms everywhere so the
/// subsampled 8x8 chroma transforms make §7.14.5 select the 6-tap filter. The
/// chroma height of 4 leaves no horizontal chroma edge, and the vertical
/// edges at x = 4, 8 and 12 see an all-100 window that any filter whose
/// weights sum to its rounding shift leaves unchanged, so only the edge at
/// x = 16 filters.
///
/// The expected samples come from §7.14.6.3's 6-tap tap lists directly:
/// output `k` is `Round2(100 * (8 - w) + 110 * w, 3)` where `w` is that row's
/// q-side weight sum, i.e. `(804 + 10 * w) >> 3`. The q-side weight sums read
/// off the spec's tap lists are 1, 3, 5 and 7.
#[test]
fn chroma_wide_deblocking_output_matches_spec_derived_vectors() {
    const WIDTH: usize = 20;
    const HEIGHT: usize = 4;
    const Q_WEIGHT_SUMS: [i32; 4] = [1, 3, 5, 7];

    let mut expected: Vec<u8> = (0..WIDTH)
        .map(|x| if x < 16 { 100u8 } else { 110u8 })
        .collect();
    for (offset, weight) in Q_WEIGHT_SUMS.into_iter().enumerate() {
        expected[14 + offset] = ((804 + 10 * weight) >> 3) as u8;
    }
    let expected: Vec<u8> = expected.repeat(HEIGHT);

    let mut grid = TxSizeGrid::new(2 * WIDTH, 2 * HEIGHT);
    for x in (0..2 * WIDTH).step_by(16) {
        grid.set_block(x, 0, 16, 16);
    }
    let params = LoopFilterParams {
        y_vertical_level: 0,
        y_horizontal_level: 0,
        u_level: 40,
        v_level: 40,
        sharpness: 0,
    };
    let results = for_each_isa(|_| {
        let chroma = || {
            let data: Vec<u8> = (0..WIDTH * HEIGHT)
                .map(|index| if index % WIDTH < 16 { 100u8 } else { 110u8 })
                .collect();
            FilterPlane::from_samples(WIDTH, HEIGHT, data, &Limits::default()).unwrap()
        };
        let mut frame = FilterFrame::new_yuv(
            plane(2 * WIDTH, 2 * HEIGHT, 0x1122),
            chroma(),
            chroma(),
            true,
            true,
        )
        .unwrap();
        deblock_frame(&mut frame, &params, Some(&grid)).unwrap();
        frame.u.unwrap().data
    });
    for (isa, data) in &results {
        assert_eq!(
            data,
            &expected,
            "{} chroma output should match the spec's 6-tap filter constants",
            isa.name()
        );
    }
}

#[test]
fn deblocking_at_frame_borders_matches_the_scalar_reference() {
    // Planes small enough that every edge position's filter window leaves the
    // plane on at least one side, and whose dimensions are not a multiple of
    // any lane count, so every run is a partial one.
    // `(9, 8)` and `(17, 12)` additionally give the vertical edges a whole
    // final chunk whose last lane is the plane's last row, which is the
    // narrow kernel's unclamped row-word path running right up against the
    // end of the sample buffer.
    for (width, height) in [
        (5usize, 5usize),
        (9, 3),
        (3, 9),
        (6, 13),
        (17, 6),
        (9, 8),
        (17, 12),
    ] {
        for tx in [false, true] {
            let grid = tx.then(|| mixed_tx_grid(width, height));
            let results = for_each_isa(|_| {
                let mut frame =
                    FilterFrame::new_monochrome(flat_blocks_plane(width, height, 0xb0_1d));
                deblock_frame(&mut frame, &deblock_params(48, 1), grid.as_ref()).unwrap();
                frame.y.data
            });
            assert_all_match(&results, "deblocked plane at the frame border");
        }
    }
}

#[test]
fn cdef_at_frame_borders_matches_the_scalar_reference() {
    // Widths and heights that are not multiples of 8 leave partial CDEF blocks
    // at the right and bottom edges, whose taps and direction search both read
    // outside the plane.
    for (width, height) in [(5usize, 5usize), (9, 3), (13, 11), (23, 17)] {
        for (primary, secondary, damping) in [(4u8, 2u8, 3u8), (15, 0, 6), (0, 4, 5)] {
            let results = for_each_isa(|_| {
                let strength = CdefStrength {
                    y_primary: primary,
                    y_secondary: secondary,
                    uv_primary: primary,
                    uv_secondary: secondary,
                    damping,
                };
                let mut frame = FilterFrame::new_monochrome(plane(width, height, 0xcdef));
                cdef_frame(&mut frame, &strength, &Limits::default()).unwrap();
                frame.y.data
            });
            assert_all_match(&results, "CDEF filtered plane at the frame border");
        }
    }
}

#[test]
fn restoration_at_region_borders_matches_the_scalar_reference() {
    // Restoration units flush against the plane's edges (so the 7-tap Wiener
    // window and the box-statistics windows clamp) and with widths that leave a
    // partial trailing vector.
    for (width, height) in [(7usize, 5usize), (11, 9), (34, 13)] {
        let results = for_each_isa(|_| {
            let mut target = plane(width, height, 0x2468);
            let unit = RestorationUnit::Wiener {
                horizontal: [3, -7, 15],
                vertical: [-2, 5, 11],
            };
            apply_restoration_unit(&mut target, &unit, 0, 0, width, height).unwrap();
            let unit = RestorationUnit::SelfGuided {
                radius: [1, 3],
                eps: [12, 30],
                weight: [40, 24],
            };
            apply_restoration_unit(&mut target, &unit, 0, 0, width, height).unwrap();
            target.data
        });
        assert_all_match(&results, "restored plane at the region border");
    }
}

// ---------------------------------------------------------------------
// Dispatch bookkeeping
// ---------------------------------------------------------------------

#[test]
fn available_isas_always_include_scalar_and_the_detected_one() {
    let isas = available_isas();
    assert!(isas.contains(&SimdIsa::Scalar));
    assert!(isas.contains(&detected_isa()));
}

#[test]
fn lane_counts_match_register_widths() {
    assert_eq!(lanes(SimdIsa::Scalar), 0);
    assert_eq!(lanes(SimdIsa::Sse41), 4);
    assert_eq!(lanes(SimdIsa::Neon), 4);
    assert_eq!(lanes(SimdIsa::Avx2), 8);
    assert!(lanes(SimdIsa::Avx2) <= MAX_LANES);
}

#[test]
fn unsupported_override_falls_back_to_scalar() {
    let _guard = lock_isa();
    let unsupported = if cfg!(target_arch = "aarch64") {
        SimdIsa::Avx2
    } else {
        SimdIsa::Neon
    };
    if !available_isas().contains(&unsupported) {
        set_active_isa(Some(unsupported));
        assert_eq!(active_isa(), SimdIsa::Scalar);
    }
    set_active_isa(None);
    assert_eq!(active_isa(), detected_isa());
}

// ---------------------------------------------------------------------
// Vector primitives
// ---------------------------------------------------------------------

/// Lane values for the packed-store checks, chosen so each vector width sees
/// negatives, both ends of `0..=255`, and magnitudes that a saturating
/// 32 -> 16 -> 8 bit pack chain would mishandle if it dropped the range clamp
/// (65536 saturates to `0xffff`, which reads back as a *negative* 16-bit lane).
const PACK_CASES: [[i32; MAX_LANES]; 3] = [
    [-300, -1, 0, 1, 128, 254, 255, 400],
    [255, 256, 65535, 65536, -1, 70000, 42, 0],
    [i32::MIN, i32::MAX, 0, 255, 1, 200, -70000, 65536],
];

/// Checks the byte-packing store and the row-word load/store pair the narrow
/// deblocking kernels use against straightforward scalar equivalents.
///
/// # Safety
/// `V`'s instruction set must be available on this host.
unsafe fn check_vector_byte_paths<V: vector::I32x>() {
    unsafe {
        for case in PACK_CASES {
            let value = V::load(&case);

            // `store_u8_clamped` must equal a per-lane `clamp(0, 255)`.
            let mut packed = [0u8; MAX_LANES];
            value.store_u8_clamped(&mut packed);
            for (lane, &input) in case.iter().enumerate().take(V::LANES) {
                assert_eq!(
                    packed[lane],
                    input.clamp(0, 255) as u8,
                    "store_u8_clamped lane {lane} of {case:?}"
                );
            }

            // ... and so must every partial count, which takes the staged path.
            for count in 0..=V::LANES {
                let mut masked = [0xaau8; MAX_LANES];
                value.store_u8_clamped_masked(&mut masked, count);
                for (lane, &input) in case.iter().enumerate().take(V::LANES) {
                    let expected = if lane < count {
                        input.clamp(0, 255) as u8
                    } else {
                        0xaa
                    };
                    assert_eq!(
                        masked[lane], expected,
                        "masked store lane {lane}, {count} of {case:?}"
                    );
                }
            }
        }

        // Row words: `LANES` rows of `stride` bytes, four of which per row are
        // the window the narrow vertical-edge filter reads and writes.
        const STRIDE: usize = 11;
        const BASE: usize = 3;
        let mut buffer: Vec<u8> = (0..STRIDE * MAX_LANES).map(|i| (i * 7 + 1) as u8).collect();
        let original = buffer.clone();

        let loaded = V::load_u32_rows(&buffer, BASE, STRIDE);
        let mut lanes = [0i32; MAX_LANES];
        loaded.store(&mut lanes);
        for (lane, &word) in lanes.iter().enumerate().take(V::LANES) {
            let at = BASE + lane * STRIDE;
            let expected =
                i32::from_le_bytes([buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]]);
            assert_eq!(word, expected, "load_u32_rows lane {lane}");
        }

        // Storing the loaded words back is the identity, and storing something
        // else touches exactly the four bytes of each row's window.
        loaded.store_u32_rows(&mut buffer, BASE, STRIDE);
        assert_eq!(buffer, original, "store_u32_rows should round-trip");

        let replacement = V::splat(0x0403_0201);
        replacement.store_u32_rows(&mut buffer, BASE, STRIDE);
        for index in 0..buffer.len() {
            let lane = index.checked_sub(BASE).map(|offset| offset / STRIDE);
            let column = index.wrapping_sub(BASE) % STRIDE;
            let inside = index >= BASE && column < 4 && lane.is_some_and(|lane| lane < V::LANES);
            let expected = if inside {
                (column + 1) as u8
            } else {
                original[index]
            };
            assert_eq!(buffer[index], expected, "store_u32_rows byte {index}");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_byte_paths_match_the_scalar_reference() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    unsafe { check_vector_byte_paths::<vector::Neon>() };
}

#[cfg(target_arch = "x86_64")]
#[test]
fn sse41_byte_paths_match_the_scalar_reference() {
    #[target_feature(enable = "sse4.1")]
    unsafe fn run() {
        unsafe { check_vector_byte_paths::<vector::Sse4>() };
    }
    if !std::arch::is_x86_feature_detected!("sse4.1") {
        return;
    }
    unsafe { run() };
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_byte_paths_match_the_scalar_reference() {
    #[target_feature(enable = "avx2")]
    unsafe fn run() {
        unsafe { check_vector_byte_paths::<vector::Avx2>() };
    }
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    unsafe { run() };
}
// ---------------------------------------------------------------------
// Forward transforms (issue #140)
// ---------------------------------------------------------------------

#[test]
fn forward_transforms_match_the_scalar_reference_at_every_size_and_type() {
    let mut rng = Lcg(0x5eed_0140_0000_0001);
    for (tx_type, sizes) in TX_TYPES {
        for &size in sizes {
            // The forward kernels stop at 32 points; AV1 defines no 64-point
            // forward transform, so skip the inverse-only 64x64 entries.
            if size > 32 {
                continue;
            }
            for _ in 0..40 {
                let residual: Vec<i32> = (0..size * size).map(|_| rng.in_range(255)).collect();
                let results = for_each_isa(|_| forward_transform(&residual, size, tx_type));
                assert_all_match(
                    &results,
                    &format!("forward {tx_type:?} {size}x{size} output"),
                );
            }
        }
    }
}

/// The same input-limit sweep the inverse transforms get: the forward bound is
/// only worth documenting if the vector kernels stay bit-exact right at it, in
/// the sign patterns that maximize each pass.
#[test]
fn forward_transforms_stay_bit_exact_at_the_documented_input_limit() {
    let mut rng = Lcg(0x5eed_0140_0000_0002);
    for (tx_type, sizes) in TX_TYPES {
        for &size in sizes {
            // The forward kernels stop at 32 points; AV1 defines no 64-point
            // forward transform, so skip the inverse-only 64x64 entries.
            if size > 32 {
                continue;
            }
            let extreme = crate::av1_encoder::transform::input_limit(size);
            for pattern in 0..6 {
                let residual: Vec<i32> = (0..size * size)
                    .map(|index| {
                        let sign = match pattern {
                            0 => 1,
                            1 => -1,
                            2 => {
                                if index % 2 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            3 => {
                                if (index / size) % 2 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            4 => {
                                if index % size < size / 2 {
                                    1
                                } else {
                                    -1
                                }
                            }
                            _ => {
                                if rng.next() & 1 == 0 {
                                    1
                                } else {
                                    -1
                                }
                            }
                        };
                        sign * extreme
                    })
                    .collect();
                let results = for_each_isa(|_| forward_transform(&residual, size, tx_type));
                assert_all_match(
                    &results,
                    &format!("forward {tx_type:?} {size}x{size} at the input limit"),
                );
            }
        }
    }
}

#[test]
fn out_of_range_forward_transform_blocks_fall_back_to_scalar() {
    for size in [4usize, 8, 16, 32] {
        let over = i64::from(crate::av1_encoder::transform::input_limit(size)) + 1;
        let residual: Vec<i32> = (0..size * size)
            .map(|index| if index == 3 { over as i32 } else { 1 })
            .collect();
        let results = for_each_isa(|isa| {
            let mut out = vec![0i32; size * size];
            let vectorized = forward_transform_simd(
                isa,
                &residual,
                size,
                Tx1d::Dct,
                Tx1d::Dct,
                false,
                false,
                &mut out,
            );
            assert!(!vectorized, "the range guard must reject this block");
            // The comparison above is only meaningful if an in-range block
            // does reach the vector kernels, so check that here too.
            let in_range = vec![1i32; size * size];
            let accepted = forward_transform_simd(
                isa,
                &in_range,
                size,
                Tx1d::Dct,
                Tx1d::Dct,
                false,
                false,
                &mut out,
            );
            assert_eq!(
                accepted,
                isa != SimdIsa::Scalar,
                "{} should{} vectorize an in-range {size}x{size} block",
                isa.name(),
                if isa == SimdIsa::Scalar { " not" } else { "" }
            );
            forward_transform(&residual, size, Av1TxType::DctDct)
        });
        assert_all_match(&results, "out-of-range forward transform output");
    }
}

/// The forward and inverse vector kernels have to agree with each other, not
/// just each with its own scalar reference: a round trip run entirely on one
/// instruction set must reproduce the residual as closely as the scalar pair.
#[test]
fn vectorized_round_trip_reproduces_the_residual() {
    let mut rng = Lcg(0x5eed_0140_0000_0003);
    for (tx_type, sizes) in TX_TYPES {
        for &size in sizes {
            // The forward kernels stop at 32 points; AV1 defines no 64-point
            // forward transform, so skip the inverse-only 64x64 entries.
            if size > 32 {
                continue;
            }
            let residual: Vec<i32> = (0..size * size).map(|_| rng.in_range(255)).collect();
            let results = for_each_isa(|_| {
                let coefficients = forward_transform(&residual, size, tx_type);
                inverse_transform(&coefficients, size, tx_type, 1, 1)
            });
            for (isa, reconstructed) in &results {
                for (&want, &got) in residual.iter().zip(reconstructed.iter()) {
                    assert!(
                        (want - i32::from(got)).abs() <= 4,
                        "{} {tx_type:?} {size}x{size} round trip: {want} became {got}",
                        isa.name()
                    );
                }
            }
            assert_all_match(&results, &format!("{tx_type:?} {size}x{size} round trip"));
        }
    }
}
