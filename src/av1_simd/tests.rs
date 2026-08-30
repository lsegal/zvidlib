//! Bit-exactness coverage: every vectorized kernel must produce output
//! identical to the scalar reference, for every instruction set the host
//! supports.
//!
//! The end-to-end filter tests drive the public filter entry points with
//! [`set_active_isa`] pinned to each instruction set in turn, which exercises
//! the dispatch conditions in `av1_filters` (run eligibility, edge handling,
//! and the scalar tail) as well as the kernels themselves. They share
//! [`ISA_LOCK`] so a pinned instruction set is not swapped out underneath a
//! test by a concurrently running one.

use std::sync::{Mutex, MutexGuard};

use super::*;
use crate::av1_encoder::wht::{fwht4x4_scalar, iwht4x4_scalar};
use crate::av1_filters::{
    CdefStrength, FilterFrame, FilterPlane, LoopFilterParams, RestorationUnit,
    apply_restoration_unit, cdef_frame, deblock_frame,
};
use crate::av1_intra::{Av1TxType, inverse_transform};
use crate::{Limits, TxSizeGrid};

static ISA_LOCK: Mutex<()> = Mutex::new(());

fn lock_isa() -> MutexGuard<'static, ()> {
    ISA_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
            let coefficients = fwht4x4(isa, &residual).expect("in-range block is vectorizable");
            assert_eq!(coefficients, fwht4x4_scalar(&residual), "{}", isa.name());
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

#[test]
fn inverse_dct_matches_the_scalar_reference_for_both_sizes() {
    let mut rng = Lcg(0x5eed_0120_0000_0002);
    for size in [4usize, 8] {
        for _ in 0..500 {
            let coefficients: Vec<i32> = (0..size * size).map(|_| rng.in_range(600)).collect();
            let results =
                for_each_isa(|_| inverse_transform(&coefficients, size, Av1TxType::DctDct, 20, 14));
            assert_all_match(&results, "inverse DCT output");
        }
    }
}

#[test]
fn out_of_range_dct_blocks_fall_back_to_scalar() {
    let coefficients: Vec<i32> = (0..16)
        .map(|index| if index == 3 { 1 << 24 } else { 1 })
        .collect();
    let results = for_each_isa(|isa| {
        let mut out = vec![0i16; 16];
        let values: Vec<i32> = coefficients.clone();
        let vectorized = inverse_dct(isa, &values, 4, &mut out);
        assert!(!vectorized, "the range guard must reject this block");
        inverse_transform(&coefficients, 4, Av1TxType::DctDct, 20, 14)
    });
    assert_all_match(&results, "out-of-range inverse DCT output");
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
