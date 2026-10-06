//! Every vector kernel against its scalar reference, on every instruction set
//! this host can run, over randomized and edge-case inputs.

use super::*;
use crate::vp8::loop_filter::filter_edge_scalar;
use crate::vp8::predict::{
    filter_block_scalar, idct_add_scalar, idct_dc_add_scalar, inverse_walsh_scalar,
    predict_subblock_scalar, tm_block_scalar,
};
use crate::vp8::tables::{BILINEAR_FILTERS, SIXTAP_FILTERS};

/// SplitMix64, so every run sees the same inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count).map(|_| self.byte()).collect()
    }

    /// A coefficient: mostly small, as real residuals are, sometimes anywhere
    /// in the 16-bit range, sometimes at its ends.
    fn coefficient(&mut self) -> i16 {
        match self.below(8) {
            0 => i16::MIN,
            1 => i16::MAX,
            2 | 3 => self.next() as i16,
            4 => 0,
            _ => (self.below(512) as i16) - 256,
        }
    }

    fn block(&mut self) -> [i16; 16] {
        std::array::from_fn(|_| self.coefficient())
    }
}

/// The vector instruction sets this host can execute.
fn vector_isas() -> Vec<SimdIsa> {
    crate::simd::available()
        .into_iter()
        .filter(|&isa| isa != SimdIsa::Scalar)
        .collect()
}

#[test]
fn every_vector_instruction_set_has_kernels() {
    for isa in vector_isas() {
        let mut plane = [0u8; 16];
        assert!(idct_dc_add(isa, 8, &mut plane, 0, 4), "{}", isa.name());
        assert_eq!(plane, [1; 16]);
    }
    assert!(!idct_dc_add(SimdIsa::Scalar, 8, &mut [0u8; 16], 0, 4));
    assert!(inverse_walsh(SimdIsa::Scalar, &[0; 16]).is_none());
}

#[test]
fn inverse_dct_matches_scalar() {
    let mut rng = Rng(1);
    for isa in vector_isas() {
        for case in 0..4000 {
            let block = match case {
                0 => [i16::MAX; 16],
                1 => [i16::MIN; 16],
                2 => std::array::from_fn(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN }),
                _ => rng.block(),
            };
            let stride = 4 + rng.below(4) as usize * 4;
            let mut expected = rng.bytes(stride * 4);
            let mut actual = expected.clone();
            idct_add_scalar(&block, &mut expected, 0, stride);
            assert!(idct_add(isa, &block, &mut actual, 0, stride));
            assert_eq!(actual, expected, "{} {block:?}", isa.name());
        }
    }
}

#[test]
fn dc_only_inverse_dct_matches_scalar_and_the_full_transform() {
    let mut rng = Rng(2);
    for isa in vector_isas() {
        for case in 0..2000 {
            let dc = match case {
                0 => i16::MAX,
                1 => i16::MIN,
                _ => rng.coefficient(),
            };
            let mut expected = rng.bytes(32);
            let mut actual = expected.clone();
            let mut full = expected.clone();
            idct_dc_add_scalar(dc, &mut expected, 4, 8);
            assert!(idct_dc_add(isa, dc, &mut actual, 4, 8));
            assert_eq!(actual, expected, "{} dc {dc}", isa.name());
            let mut block = [0i16; 16];
            block[0] = dc;
            idct_add_scalar(&block, &mut full, 4, 8);
            assert_eq!(full, expected, "dc {dc}");
        }
    }
}

#[test]
fn inverse_walsh_matches_scalar() {
    let mut rng = Rng(3);
    for isa in vector_isas() {
        for case in 0..4000 {
            let block = match case {
                0 => [i16::MAX; 16],
                1 => [i16::MIN; 16],
                _ => rng.block(),
            };
            assert_eq!(
                inverse_walsh(isa, &block),
                Some(inverse_walsh_scalar(&block)),
                "{} {block:?}",
                isa.name()
            );
        }
    }
}

#[test]
fn sub_pixel_filters_match_scalar() {
    let mut rng = Rng(4);
    for isa in vector_isas() {
        for filters in [&SIXTAP_FILTERS, &BILINEAR_FILTERS] {
            for size in [4, 8, 16] {
                for fraction_x in 0..8 {
                    for fraction_y in 0..8 {
                        for case in 0..6 {
                            let window_width = size + 5;
                            let window: Vec<u8> = match case {
                                // Alternating extremes drive the six-tap
                                // filter's negative taps past both clamps.
                                0 => (0..21 * 21)
                                    .map(|i| {
                                        if (i + i / window_width) % 2 == 0 {
                                            0
                                        } else {
                                            255
                                        }
                                    })
                                    .collect(),
                                1 => vec![255; 21 * 21],
                                _ => rng.bytes(21 * 21),
                            };
                            let stride = 32;
                            let destination = 3 * stride + 8;
                            let mut expected = rng.bytes(stride * 20);
                            let mut actual = expected.clone();
                            let (horizontal, vertical) =
                                (&filters[fraction_x], &filters[fraction_y]);
                            filter_block_scalar(
                                &window,
                                window_width,
                                size,
                                size,
                                horizontal,
                                vertical,
                                &mut expected,
                                destination,
                                stride,
                            );
                            assert!(filter_block(
                                isa,
                                &window,
                                window_width,
                                size,
                                size,
                                horizontal,
                                vertical,
                                &mut actual,
                                destination,
                                stride,
                            ));
                            assert_eq!(
                                actual,
                                expected,
                                "{} {size}x{size} fraction ({fraction_x}, {fraction_y})",
                                isa.name()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tm_prediction_matches_scalar() {
    let mut rng = Rng(5);
    for isa in vector_isas() {
        for n in [8, 16] {
            for case in 0..500 {
                let (above, left) = match case {
                    0 => (vec![0; n + 1], vec![255; n]),
                    1 => {
                        let mut above = vec![255; n + 1];
                        above[0] = 0;
                        (above, vec![0; n])
                    }
                    _ => (rng.bytes(n + 1), rng.bytes(n)),
                };
                let stride = 24;
                let mut expected = rng.bytes(stride * n + 8);
                let mut actual = expected.clone();
                tm_block_scalar(&above, &left, &mut expected, 4, stride);
                assert!(tm_block(isa, &above, &left, &mut actual, 4, stride));
                assert_eq!(actual, expected, "{} {n}x{n}", isa.name());
            }
        }
    }
}

#[test]
fn subblock_prediction_matches_scalar_in_every_mode() {
    let mut rng = Rng(6);
    for isa in vector_isas() {
        for mode in 0..10 {
            for case in 0..500 {
                let (above, left): ([u8; 9], [u8; 4]) = match case {
                    0 => ([0; 9], [255; 4]),
                    1 => ([255; 9], [0; 4]),
                    _ => (
                        std::array::from_fn(|_| rng.byte()),
                        std::array::from_fn(|_| rng.byte()),
                    ),
                };
                let stride = 12;
                let mut expected = rng.bytes(stride * 4 + 4);
                let mut actual = expected.clone();
                predict_subblock_scalar(mode, &above, &left, &mut expected, 2, stride);
                assert!(subblock(isa, mode, &above, &left, &mut actual, 2, stride));
                assert_eq!(
                    actual,
                    expected,
                    "{} mode {mode} above {above:?} left {left:?}",
                    isa.name()
                );
            }
        }
    }
}

/// An area around an edge whose samples vary by at most `spread` from a
/// common base, so that some segments pass the filter thresholds and some do
/// not, as in real content.
fn edge_area(rng: &mut Rng, len: usize, spread: u64) -> Vec<u8> {
    let base = rng.below(256) as i32;
    (0..len)
        .map(|_| {
            let offset = rng.below(spread * 2 + 1) as i32 - spread as i32;
            (base + offset).clamp(0, 255) as u8
        })
        .collect()
}

#[test]
fn loop_filter_edges_match_scalar() {
    let mut rng = Rng(7);
    let stride = 40;
    for isa in vector_isas() {
        for case in 0..3000 {
            let count = if rng.below(2) == 0 { 8 } else { 16 };
            let vertical = rng.below(2) == 0;
            let macroblock_edge = rng.below(2) == 0;
            let simple = rng.below(4) == 0;
            let thresholds = EdgeThresholds {
                edge_limit: rng.below(140) as i32,
                interior: 1 + rng.below(63) as i32,
                hev_threshold: rng.below(4) as i32,
            };
            let spread = [0, 2, 8, 30, 128][rng.below(5) as usize];
            let mut expected = match case % 7 {
                0 => rng.bytes(stride * 24),
                1 => (0..stride * 24)
                    .map(|i| if i % 2 == 0 { 0 } else { 255 })
                    .collect(),
                _ => edge_area(&mut rng, stride * 24, spread),
            };
            let mut actual = expected.clone();
            let at = 4 * stride + 4;
            let (step, advance) = if vertical { (1, stride) } else { (stride, 1) };
            filter_edge_scalar(
                &mut expected,
                at,
                step,
                advance,
                count,
                macroblock_edge,
                thresholds,
                simple,
            );
            assert!(filter_edge(
                isa,
                &mut actual,
                at,
                step,
                stride,
                count,
                macroblock_edge,
                thresholds,
                simple,
            ));
            assert_eq!(
                actual,
                expected,
                "{} vertical {vertical} count {count} macroblock {macroblock_edge} \
                 simple {simple} {thresholds:?}",
                isa.name()
            );
        }
    }
}

/// The site follows the crate-wide override and falls back to detection.
#[test]
fn the_dispatch_site_follows_the_override() {
    let _guard = crate::simd::test_lock();
    for isa in crate::simd::available() {
        crate::simd::set_override(Some(isa));
        assert_eq!(active_isa(), isa);
    }
    crate::simd::set_override(None);
    assert_eq!(active_isa(), crate::simd::detected());
}
