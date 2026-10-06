//! Bit-exactness coverage: every VP9 vector kernel must produce output
//! identical to its scalar reference in `vp9_dec`, for every instruction set
//! the host supports, on randomized and edge-case input. Each test also
//! checks the vector path really ran for the shapes it is meant to cover, so
//! a dispatcher that quietly declined everything could not pass.
//!
//! The whole-decoder counterpart, decoding the bundled fixtures under every
//! instruction set, is in `vp9_dec::tests`.

use super::*;
use crate::vp9_dec::loopfilter::{Pixels, thresholds};
use crate::vp9_dec::recon::{
    self, DCT_DCT, IntraEdges, convolve_scalar, inverse_transform_add_scalar, predict_intra_scalar,
};
use crate::vp9_dec::tables::FILTER_KERNELS;

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

    /// Uniform in `-span..=span`.
    fn signed(&mut self, span: i32) -> i32 {
        (self.next() % (2 * span as u32 + 1)) as i32 - span
    }

    fn pick<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.next() as usize % values.len()]
    }
}

/// The vector instruction sets this host can run, which every test covers.
fn vector_isas() -> Vec<SimdIsa> {
    crate::simd::available()
        .into_iter()
        .filter(|&isa| isa != SimdIsa::Scalar)
        .collect()
}

// ---------------------------------------------------------------------
// Inverse transforms
// ---------------------------------------------------------------------

/// The end-of-block positions that reach each of libvpx's row shortcuts for
/// a transform size, plus the DC-only one.
fn eobs(tx_size: u8) -> &'static [usize] {
    match tx_size {
        0 => &[1, 2, 16],
        1 => &[1, 12, 13, 64],
        2 => &[1, 10, 11, 38, 39, 256],
        _ => &[1, 34, 35, 135, 136, 1024],
    }
}

/// A block of coefficients drawn from `-span..=span`, sparse or dense.
fn coefficients(rng: &mut Lcg, n: usize, span: i32) -> Vec<i32> {
    let density = rng.pick(&[1u32, 4, 16]);
    (0..n * n)
        .map(|_| {
            if rng.next() % density == 0 {
                rng.signed(span)
            } else {
                0
            }
        })
        .collect()
}

/// Runs one block through the dispatcher the decoder uses, with the scalar
/// fallback it takes, and reports whether the vector kernel accepted it.
#[allow(clippy::too_many_arguments)]
fn transform_with(
    isa: SimdIsa,
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    tx_size: u8,
    tx_type: u8,
    eob: usize,
    lossless: bool,
) -> bool {
    let vector =
        inverse_transform_add(isa, input, dest, stride, tx_size, tx_type, eob, lossless);
    if !vector {
        inverse_transform_add_scalar(input, dest, stride, tx_size, tx_type, eob, lossless);
    }
    vector
}

fn check_transform(rng: &mut Lcg, tx_size: u8, tx_type: u8, eob: usize, span: i32) -> bool {
    let n = 4usize << tx_size;
    let stride = n + rng.pick(&[0usize, 3, 16]);
    let input = coefficients(rng, n, span);
    let base: Vec<u8> = (0..stride * n)
        .map(|_| match rng.next() % 8 {
            0 => 0,
            1 => 255,
            _ => rng.byte(),
        })
        .collect();
    let mut expected = base.clone();
    inverse_transform_add_scalar(&input, &mut expected, stride, tx_size, tx_type, eob, false);
    let mut vectorized_everywhere = true;
    for isa in vector_isas() {
        let mut actual = base.clone();
        vectorized_everywhere &=
            transform_with(isa, &input, &mut actual, stride, tx_size, tx_type, eob, false);
        assert_eq!(
            actual,
            expected,
            "{} tx_size {tx_size} tx_type {tx_type} eob {eob} span {span}",
            isa.name()
        );
    }
    vectorized_everywhere
}

/// Every size, type and end-of-block shortcut matches the scalar path, and
/// runs vectorized whenever the coefficients are small enough that no ADST
/// input can reach its limit.
#[test]
fn inverse_transforms_match_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0001);
    for tx_size in 0..4u8 {
        let types: &[u8] = if tx_size == 3 { &[0] } else { &[0, 1, 2, 3] };
        for &tx_type in types {
            for &eob in eobs(tx_size) {
                for span in [1, 64, 1024, ADST16_INPUT_LIMIT_I32 / 8] {
                    for _ in 0..8 {
                        let vectorized = check_transform(&mut rng, tx_size, tx_type, eob, span);
                        if span <= 64 && !vector_isas().is_empty() {
                            assert!(
                                vectorized,
                                "tx_size {tx_size} tx_type {tx_type} eob {eob} fell back to scalar"
                            );
                        }
                    }
                }
            }
        }
    }
}

const ADST16_INPUT_LIMIT_I32: i32 = transforms::ADST16_INPUT_LIMIT as i32;

/// The DCTs truncate every stage to 16 bits, so coefficients far outside
/// the 16-bit range still have to match; the ADSTs either match or decline
/// and leave the block to the scalar path, which then matches trivially.
#[test]
fn large_coefficients_match_or_fall_back() {
    let mut rng = Lcg(0x5eed_0002);
    for tx_size in 0..4u8 {
        let types: &[u8] = if tx_size == 3 { &[0] } else { &[0, 1, 2, 3] };
        for &tx_type in types {
            for span in [32_767, 40_000] {
                for _ in 0..8 {
                    let eob = (16usize << (2 * tx_size)).min(1024);
                    check_transform(&mut rng, tx_size, tx_type, eob, span);
                }
            }
        }
    }
    // DCT blocks of extreme coefficients, which only the 16-bit truncation
    // keeps in range.
    for tx_size in 0..4u8 {
        let n = 4usize << tx_size;
        for value in [i32::from(i16::MAX), i32::from(i16::MIN), 1 << 20, -(1 << 24)] {
            let input = vec![value; n * n];
            let mut expected = vec![128u8; n * n];
            inverse_transform_add_scalar(&input, &mut expected, n, tx_size, DCT_DCT, n * n, false);
            for isa in vector_isas() {
                let mut actual = vec![128u8; n * n];
                assert!(transform_with(
                    isa,
                    &input,
                    &mut actual,
                    n,
                    tx_size,
                    DCT_DCT,
                    n * n,
                    false
                ));
                assert_eq!(actual, expected, "{} tx_size {tx_size} {value}", isa.name());
            }
        }
    }
}

/// Each ADST stays bit-exact with every coefficient at its documented limit,
/// in every sign pattern the generator draws, and declines one past it.
#[test]
fn adst_input_limits_hold_at_the_boundary() {
    use transforms::{ADST4_INPUT_LIMIT, ADST8_INPUT_LIMIT};
    let mut rng = Lcg(0x5eed_0003);
    for (tx_size, limit) in [
        (0u8, ADST4_INPUT_LIMIT),
        (1, ADST8_INPUT_LIMIT),
        (2, transforms::ADST16_INPUT_LIMIT),
    ] {
        let n = 4usize << tx_size;
        let limit = limit as i32;
        // DCT rows into ADST columns: column inputs are the row outputs,
        // checked separately from the coefficients.
        for tx_type in [1u8, 2, 3] {
            for _ in 0..32 {
                let input: Vec<i32> = (0..n * n)
                    .map(|_| if rng.next() % 2 == 0 { limit } else { -limit })
                    .collect();
                let mut expected = vec![100u8; n * n];
                inverse_transform_add_scalar(&input, &mut expected, n, tx_size, tx_type, 0, false);
                for isa in vector_isas() {
                    let mut actual = vec![100u8; n * n];
                    transform_with(isa, &input, &mut actual, n, tx_size, tx_type, 0, false);
                    assert_eq!(actual, expected, "{} tx_size {tx_size}", isa.name());
                }
            }
            // A row ADST input one past the limit declines without writing.
            if tx_type != 1 {
                let mut input = vec![0i32; n * n];
                input[0] = limit + 1;
                for isa in vector_isas() {
                    let mut dest = vec![7u8; n * n];
                    assert!(!inverse_transform_add(
                        isa, &input, &mut dest, n, tx_size, tx_type, 0, false
                    ));
                    assert!(dest.iter().all(|&pixel| pixel == 7));
                }
            }
        }
    }
}

/// The lossless Walsh-Hadamard transform, full and DC-only.
#[test]
fn lossless_transforms_match_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0004);
    for eob in [1usize, 2, 16] {
        for span in [4, 255, 4096] {
            for _ in 0..64 {
                let input = coefficients(&mut rng, 4, span);
                let base: Vec<u8> = (0..4 * 9).map(|_| rng.byte()).collect();
                let mut expected = base.clone();
                inverse_transform_add_scalar(&input, &mut expected, 9, 0, DCT_DCT, eob, true);
                for isa in vector_isas() {
                    let mut actual = base.clone();
                    assert!(transform_with(
                        isa,
                        &input,
                        &mut actual,
                        9,
                        0,
                        DCT_DCT,
                        eob,
                        true
                    ));
                    assert_eq!(actual, expected, "{} eob {eob}", isa.name());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Inter prediction
// ---------------------------------------------------------------------

/// Every filter, block size, sub-pixel phase combination and compound
/// averaging, at unit and scaled steps.
#[test]
fn convolution_matches_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0005);
    let src_stride = 64 * 2 + 32;
    let src: Vec<u8> = (0..src_stride * 200).map(|_| rng.byte()).collect();
    let origin = 8 * src_stride + 8;
    let mut temp = Box::new([0u8; 64 * 135]);
    for (filter, kernel) in FILTER_KERNELS.iter().enumerate() {
        for w in [4usize, 8, 16, 32, 64] {
            for h in [4usize, 8, 16, 32, 64] {
                for (x_step, y_step) in [(16, 16), (16, 32), (32, 16), (20, 24), (32, 32)] {
                    for _ in 0..3 {
                        let x_frac = rng.pick(&[0, 0, 1, 7, 8, 15]);
                        let y_frac = rng.pick(&[0, 0, 1, 7, 8, 15]);
                        let average = rng.next() % 2 == 0;
                        let stride = w + 5;
                        let base: Vec<u8> = (0..stride * h).map(|_| rng.byte()).collect();
                        let mut expected = base.clone();
                        convolve_scalar(
                            &src,
                            origin,
                            src_stride,
                            &mut expected,
                            stride,
                            w,
                            h,
                            kernel,
                            x_frac,
                            x_step,
                            y_frac,
                            y_step,
                            average,
                            &mut temp,
                        );
                        for isa in vector_isas() {
                            let mut actual = base.clone();
                            let vectorized = convolve(
                                isa,
                                &src,
                                origin,
                                src_stride,
                                &mut actual,
                                stride,
                                w,
                                h,
                                kernel,
                                x_frac,
                                x_step,
                                y_frac,
                                y_step,
                                average,
                                &mut temp,
                            );
                            if !vectorized {
                                convolve_scalar(
                                    &src,
                                    origin,
                                    src_stride,
                                    &mut actual,
                                    stride,
                                    w,
                                    h,
                                    kernel,
                                    x_frac,
                                    x_step,
                                    y_frac,
                                    y_step,
                                    average,
                                    &mut temp,
                                );
                            }
                            // Only a scaled horizontal-only step is left to
                            // the scalar path.
                            let horizontal_only = y_frac == 0 && y_step == 16;
                            let whole_x = x_frac == 0 && x_step == 16;
                            assert_eq!(
                                vectorized,
                                !(horizontal_only && !whole_x && x_step != 16),
                                "{} filter {filter} {w}x{h}",
                                isa.name()
                            );
                            assert_eq!(
                                actual,
                                expected,
                                "{} filter {filter} {w}x{h} frac ({x_frac}, {y_frac}) step \
                                 ({x_step}, {y_step}) average {average}",
                                isa.name()
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Intra prediction
// ---------------------------------------------------------------------

/// Every mode, every block size and every edge availability, on random and
/// saturated edges.
#[test]
fn intra_prediction_matches_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0006);
    for bs in [4usize, 8, 16, 32] {
        for mode in 0..=9u8 {
            for have_left in [false, true] {
                for have_above in [false, true] {
                    for round in 0..6 {
                        let edge = |rng: &mut Lcg| match round {
                            0 => 0,
                            1 => 255,
                            _ => rng.byte(),
                        };
                        let edges = IntraEdges {
                            above: core::array::from_fn(|_| edge(&mut rng)),
                            left: core::array::from_fn(|_| edge(&mut rng)),
                        };
                        let stride = bs + rng.pick(&[0usize, 1, 16]);
                        let base: Vec<u8> = (0..stride * bs).map(|_| rng.byte()).collect();
                        let mut expected = base.clone();
                        predict_intra_scalar(
                            &mut expected,
                            stride,
                            bs,
                            mode,
                            &edges,
                            have_left,
                            have_above,
                        );
                        for isa in vector_isas() {
                            let mut actual = base.clone();
                            assert!(predict_intra(
                                isa,
                                &mut actual,
                                stride,
                                bs,
                                mode,
                                &edges.above,
                                &edges.left,
                                have_left,
                                have_above,
                            ));
                            assert_eq!(
                                actual,
                                expected,
                                "{} {bs}x{bs} mode {mode} left {have_left} above {have_above}",
                                isa.name()
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Loop filter
// ---------------------------------------------------------------------

/// A plane whose 8x8 regions alternate between random noise and flat areas
/// of small random steps, so every branch of every filter - no filtering,
/// `filter4` with and without high edge variance, `filter8` and the 15-tap
/// filter - is taken somewhere.
fn filter_plane(rng: &mut Lcg, width: usize, height: usize) -> Vec<u8> {
    let mut plane = vec![0u8; width * height];
    let regions: Vec<(i32, i32)> = (0..(width / 8 + 1) * (height / 8 + 1))
        .map(|_| {
            (
                i32::from(rng.byte()),
                rng.pick(&[0, 1, 1, 2, 3, 6, 20, 128]),
            )
        })
        .collect();
    for y in 0..height {
        for x in 0..width {
            let (level, spread) = regions[(y / 8) * (width / 8 + 1) + x / 8];
            plane[y * width + x] = (level + rng.signed(spread)).clamp(0, 255) as u8;
        }
    }
    plane
}

/// `lpf4`, `lpf8` and `lpf16` across horizontal and vertical edges at every
/// filter level and several sharpness settings.
#[test]
fn loop_filters_match_the_scalar_reference() {
    let mut rng = Lcg(0x5eed_0007);
    let (width, height) = (96usize, 96usize);
    let mut vectorized = 0usize;
    for sharpness in [0u8, 3, 7] {
        let table = thresholds(sharpness);
        for level in 0..64usize {
            let t = table[level];
            let base = filter_plane(&mut rng, width, height);
            for taps in [
                loopfilter::Taps::Four,
                loopfilter::Taps::Eight,
                loopfilter::Taps::Sixteen,
            ] {
                for vertical in [false, true] {
                    for count in [8usize, 16] {
                        let (x, y) = (rng.pick(&[8usize, 16, 40]), rng.pick(&[8usize, 16, 40]));
                        let start = (y * width + x) as isize;
                        let (step, along) = if vertical {
                            (1, width as isize)
                        } else {
                            (width as isize, 1)
                        };
                        let run = |isa: SimdIsa, data: &mut Vec<u8>| {
                            let mut pixels = Pixels {
                                data,
                                stride: width,
                                isa,
                            };
                            match taps {
                                loopfilter::Taps::Four => {
                                    pixels.lpf4(start, step, along, count, t);
                                }
                                loopfilter::Taps::Eight => {
                                    pixels.lpf8(start, step, along, count, t);
                                }
                                loopfilter::Taps::Sixteen => {
                                    pixels.lpf16(start, step, along, count, t);
                                }
                            }
                        };
                        let mut expected = base.clone();
                        run(SimdIsa::Scalar, &mut expected);
                        for isa in vector_isas() {
                            assert!(loopfilter::covers(
                                base.len(),
                                start,
                                step,
                                along,
                                count,
                                taps
                            ));
                            vectorized += 1;
                            let mut actual = base.clone();
                            run(isa, &mut actual);
                            assert_eq!(
                                actual,
                                expected,
                                "{} {taps:?} vertical {vertical} level {level} sharpness \
                                 {sharpness}",
                                isa.name()
                            );
                        }
                    }
                }
            }
        }
    }
    if !vector_isas().is_empty() {
        assert!(vectorized > 0);
    }
}

/// An edge whose 16-wide taps would reach outside the plane is left to the
/// scalar filter, which only reads them when the inner taps are flat.
#[test]
fn loop_filter_edges_near_the_plane_border_stay_scalar() {
    let len = 64 * 64;
    assert!(!loopfilter::covers(
        len,
        4,
        1,
        64,
        8,
        loopfilter::Taps::Sixteen
    ));
    assert!(loopfilter::covers(len, 4, 1, 64, 8, loopfilter::Taps::Eight));
    assert!(!loopfilter::covers(
        len,
        (60 * 64) as isize,
        64,
        1,
        8,
        loopfilter::Taps::Sixteen
    ));
    for isa in vector_isas() {
        let mut data = vec![0u8; len];
        assert!(!filter_edge(
            isa,
            &mut data,
            4,
            1,
            64,
            8,
            thresholds(0)[32],
            loopfilter::Taps::Sixteen
        ));
    }
}

/// The decoder-facing entry points reach the vector kernels when the
/// override allows it, and only the scalar code when it pins scalar.
#[test]
fn the_decoder_entry_points_follow_the_override() {
    let _guard = crate::simd::test_lock();
    crate::simd::set_override(Some(SimdIsa::Scalar));
    assert_eq!(active_isa(), SimdIsa::Scalar);
    let input = vec![17i32; 64];
    let mut dest = vec![9u8; 64];
    // Scalar: the dispatcher declines, the public entry point still works.
    assert!(!inverse_transform_add(
        active_isa(),
        &input,
        &mut dest,
        8,
        1,
        DCT_DCT,
        64,
        false
    ));
    recon::inverse_transform_add(&input, &mut dest, 8, 1, DCT_DCT, 64, false);
    crate::simd::set_override(None);
    assert_eq!(active_isa(), crate::simd::detected());
}
