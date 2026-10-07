//! Every VP8 vector kernel against its scalar reference.
//!
//! Each test runs the encoder's or the decoder's own entry point once with the
//! crate-wide override pinned to scalar, which is the reference, and once per
//! vector instruction set this host has, and requires identical results. The
//! inputs are random, plus the extremes each kernel's arithmetic is most
//! likely to get wrong: saturated samples, full-range `i16` coefficients and
//! frame edges.

use super::super::frame_encoder::{
    forward_walsh, quantize, residual_dct, sad16, sad16_full, satd, satd4,
};
use super::super::loop_filter::{FrameFilter, MacroblockFilter, filter_frame};
use super::super::predict::{
    Edges, Plane, idct_add, idct_dc_add, idct_dc_add_row, idct_dc_add_row_scalar, inverse_walsh,
    predict_block, predict_inter, predict_subblock, predict_subblock_scalar,
};
use super::super::tables::{AC_Q_LOOKUP, BILINEAR_FILTERS, DC_Q_LOOKUP, SIXTAP_FILTERS, TM_PRED};
use crate::simd::{self, SimdIsa};

/// A small deterministic generator, so a failure reproduces.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next() % bound
    }

    fn sample(&mut self) -> u8 {
        self.next() as u8
    }

    fn coefficient(&mut self) -> i16 {
        self.next() as i16
    }

    fn plane(&mut self, width: usize, height: usize) -> Plane {
        let mut plane = Plane::new(width, height);
        for sample in &mut plane.data {
            *sample = self.sample();
        }
        plane
    }

    /// A plane of smooth gradients with a little noise and occasional steps:
    /// the content the loop filter actually filters, rather than noise it
    /// leaves alone.
    fn smooth_plane(&mut self, width: usize, height: usize) -> Plane {
        let mut plane = Plane::new(width, height);
        let step = self.below(60) as i32;
        let noise = 1 + self.below(12) as i32;
        for y in 0..height {
            for x in 0..width {
                let mut value = 60 + (x as i32 * 3 + y as i32 * 2) % 120;
                if (x / 8 + y / 8) % 2 == 1 {
                    value += step;
                }
                value += self.below(noise as u32) as i32 - noise / 2;
                plane.data[y * width + x] = value.clamp(0, 255) as u8;
            }
        }
        plane
    }
}

/// Runs `run` pinned to scalar and then to every vector instruction set this
/// host has, and requires every result to equal the scalar one.
fn across_isas<T: PartialEq + std::fmt::Debug>(label: &str, mut run: impl FnMut() -> T) {
    let _guard = simd::test_lock();
    simd::set_override(Some(SimdIsa::Scalar));
    let reference = run();
    for isa in simd::available() {
        if isa == SimdIsa::Scalar {
            continue;
        }
        simd::set_override(Some(isa));
        let actual = run();
        if actual != reference {
            simd::set_override(None);
            panic!(
                "{label}: {} diverged from scalar\n  scalar: {reference:?}\n  {}: {actual:?}",
                isa.name(),
                isa.name()
            );
        }
    }
    simd::set_override(None);
}

/// Every vector instruction set reaches a vector kernel rather than quietly
/// falling back to scalar, which would make the comparisons above vacuous.
#[test]
fn every_vector_instruction_set_takes_the_vector_kernels() {
    let _guard = simd::test_lock();
    let plane = Plane::new(16, 16);
    let block = [0i16; 16];
    let window = [0u8; 21 * 21];
    for isa in simd::available() {
        simd::set_override(Some(isa));
        let vector = isa != SimdIsa::Scalar;
        assert_eq!(super::encode_isa(), isa);
        assert_eq!(super::recon_isa(), isa);
        assert_eq!(super::decode_isa(), isa);
        assert_eq!(
            super::sad16(&plane.data, 16, &plane.data, 16).is_some(),
            vector
        );
        assert_eq!(super::satd4(&block).is_some(), vector);
        assert_eq!(
            super::satd(&plane.data, &plane.data, 0, 16, 16).is_some(),
            vector
        );
        assert_eq!(
            super::residual_dct(&plane.data, &plane.data, 0, 16).is_some(),
            vector
        );
        assert_eq!(super::forward_walsh(&block).is_some(), vector);
        assert_eq!(super::quantize(&block, [8, 8], 0).is_some(), vector);
        assert_eq!(super::inverse_walsh(&block).is_some(), vector);
        let mut output = [0u8; 256];
        assert_eq!(super::idct_add(&block, &mut output, 0, 16), vector);
        for width in [4, 8, 16] {
            assert_eq!(
                super::sixtap(
                    &window,
                    width,
                    width,
                    &SIXTAP_FILTERS[2],
                    &SIXTAP_FILTERS[5],
                    &mut output,
                    0,
                    16
                ),
                vector,
                "width {width}"
            );
        }
        assert_eq!(
            super::tm_predict(&[0; 16], &[0; 16], 0, 16, &mut output, 0, 16),
            vector
        );
        for mode in 0..10 {
            assert_eq!(
                super::subblock(mode, &[0; 9], &[0; 4], &mut output, 0, 16),
                vector,
                "subblock mode {mode}"
            );
        }
        assert_eq!(super::idct_dc_add_row(&[8; 4], &mut output, 0, 16), vector);
    }
    simd::set_override(None);
}

#[test]
fn subblock_prediction_matches_scalar_in_every_mode() {
    let mut random = Random(0x568);
    for mode in 0..10 {
        for round in 0..256 {
            let (above, left): ([u8; 9], [u8; 4]) = match round {
                0 => ([0; 9], [255; 4]),
                1 => ([255; 9], [0; 4]),
                2 => {
                    let mut above = [255; 9];
                    above[0] = 0;
                    (above, [255; 4])
                }
                _ => (
                    std::array::from_fn(|_| random.sample()),
                    std::array::from_fn(|_| random.sample()),
                ),
            };
            // The vector kernel against the scalar function directly, as well
            // as through the entry point the decoder calls.
            let mut expected = vec![0x5a; 12 * 4 + 4];
            predict_subblock_scalar(mode, &above, &left, &mut expected, 2, 12);
            across_isas(&format!("subblock mode {mode}"), || {
                let mut plane = vec![0x5a; 12 * 4 + 4];
                predict_subblock(mode, &above, &left, &mut plane, 2, 12);
                assert_eq!(plane, expected, "mode {mode} above {above:?} left {left:?}");
                plane
            });
        }
    }
}

#[test]
fn dc_only_inverse_dct_matches_scalar_and_the_full_transform() {
    let mut random = Random(0xdc);
    let mut dc = |round: usize| match round {
        0 => i16::MAX,
        1 => i16::MIN,
        2 => 0,
        _ => random.coefficient() >> random.below(12),
    };
    let mut samples = Random(0x5a);
    for round in 0..512 {
        for blocks in [2, 4] {
            let dcs: Vec<i16> = (0..blocks).map(|block| dc(round + block)).collect();
            let stride = 24;
            let mut plane = vec![0u8; stride * 6];
            for sample in &mut plane {
                *sample = match round % 3 {
                    0 => samples.sample(),
                    1 => 255,
                    _ => 0,
                };
            }
            let offset = stride + 3;
            // The shortcut is exactly the full inverse DCT of blocks whose
            // only coefficient is the DC.
            let mut full = plane.clone();
            let mut shortcut = plane.clone();
            {
                let _guard = simd::test_lock();
                simd::set_override(Some(SimdIsa::Scalar));
                for (index, &dc) in dcs.iter().enumerate() {
                    let mut block = [0i16; 16];
                    block[0] = dc;
                    idct_add(&block, &mut full, offset + index * 4, stride);
                    idct_dc_add(dc, &mut shortcut, offset + index * 4, stride);
                }
                simd::set_override(None);
            }
            assert_eq!(shortcut, full, "dcs {dcs:?}");
            let mut reference = plane.clone();
            idct_dc_add_row_scalar(&dcs, &mut reference, offset, stride);
            assert_eq!(reference, full, "dcs {dcs:?}");
            across_isas(&format!("dc-only idct row {dcs:?}"), || {
                let mut output = plane.clone();
                idct_dc_add_row(&dcs, &mut output, offset, stride);
                assert_eq!(output, full, "dcs {dcs:?}");
                output
            });
        }
    }
}

#[test]
fn sad_matches_scalar() {
    let mut random = Random(0x5ad);
    for round in 0..64 {
        let mut source = random.plane(48, 48);
        let mut reference = random.plane(48, 48);
        if round == 0 {
            source.data.fill(255);
            reference.data.fill(0);
        }
        let (x0, y0) = (random.below(3) as usize * 16, random.below(3) as usize * 16);
        let origin = y0 * 48 + x0;
        across_isas("sad16", || sad16(&source, &reference, origin));
        // Displacements inside the reference take the vector kernel, and ones
        // that cross its edge the clamped scalar path; both must agree.
        for _ in 0..8 {
            let dx = random.below(81) as i32 - 40;
            let dy = random.below(81) as i32 - 40;
            across_isas("sad16_full", || {
                sad16_full(&source, &reference, x0, y0, dx, dy)
            });
        }
    }
}

#[test]
fn satd_matches_scalar() {
    let mut random = Random(0x5a7d);
    for round in 0..64 {
        let source = random.plane(32, 32);
        let prediction = if round == 0 {
            Plane {
                data: source.data.iter().map(|&value| 255 - value).collect(),
                ..source.clone()
            }
        } else {
            random.plane(32, 32)
        };
        let origin = random.below(2) as usize * 16 * 32 + random.below(2) as usize * 16;
        across_isas("satd16", || satd::<16>(&source, &prediction, origin));
        across_isas("satd8", || satd::<8>(&source, &prediction, origin));
        across_isas("satd4 plane", || satd::<4>(&source, &prediction, origin));

        let mut block = [0i16; 16];
        for value in &mut block {
            *value = if round < 8 {
                random.coefficient()
            } else {
                random.below(511) as i16 - 255
            };
        }
        across_isas("satd4", || satd4(&block));
    }
    across_isas("satd4 extremes", || {
        [i16::MIN, i16::MAX, -255, 255].map(|value| satd4(&[value; 16]))
    });
}

#[test]
fn forward_transforms_match_scalar() {
    let mut random = Random(0xfdc7);
    for round in 0..256 {
        let mut source = random.plane(16, 16);
        let mut prediction = random.plane(16, 16);
        if round == 0 {
            source.data.fill(255);
            prediction.data.fill(0);
        } else if round == 1 {
            source.data.fill(0);
            prediction.data.fill(255);
        }
        let offset = random.below(4) as usize * 4 * 16 + random.below(4) as usize * 4;
        across_isas("residual_dct", || {
            residual_dct(&source, &prediction, offset)
        });

        let mut dc = [0i16; 16];
        for value in &mut dc {
            *value = if round < 16 {
                random.coefficient()
            } else {
                random.below(8161) as i16 - 4080
            };
        }
        across_isas("forward_walsh", || forward_walsh(&dc));
    }
    across_isas("forward_walsh extremes", || {
        [i16::MIN, i16::MAX, 0, -1, 1].map(|value| forward_walsh(&[value; 16]))
    });
}

#[test]
fn quantization_matches_scalar() {
    let mut random = Random(0x9a47);
    for round in 0..512 {
        let q = random.below(128) as usize;
        let factors = match round % 3 {
            0 => [DC_Q_LOOKUP[q], AC_Q_LOOKUP[q]],
            1 => [DC_Q_LOOKUP[q] * 2, (AC_Q_LOOKUP[q] * 155 / 100).max(8)],
            _ => [DC_Q_LOOKUP[q].min(132), AC_Q_LOOKUP[q]],
        };
        let mut coefficients = [0i16; 16];
        for value in &mut coefficients {
            *value = match round % 4 {
                0 => random.coefficient(),
                1 => random.below(65) as i16 - 32,
                _ => random.below(4097) as i16 - 2048,
            };
        }
        if round == 0 {
            coefficients = [i16::MIN; 16];
        } else if round == 1 {
            coefficients = [i16::MAX; 16];
        }
        for first in [0, 1] {
            across_isas("quantize", || quantize(&coefficients, factors, first));
        }
    }
}

#[test]
fn inverse_transforms_match_scalar() {
    let mut random = Random(0x1d37);
    for round in 0..256 {
        let mut coefficients = [0i16; 16];
        for value in &mut coefficients {
            *value = if round < 64 {
                random.coefficient()
            } else {
                random.below(4097) as i16 - 2048
            };
        }
        if round == 0 {
            coefficients = [i16::MIN; 16];
        } else if round == 1 {
            coefficients = [i16::MAX; 16];
        }
        let plane = random.plane(16, 16);
        let offset = random.below(4) as usize * 4 * 16 + random.below(4) as usize * 4;
        across_isas("idct_add", || {
            let mut plane = plane.data.clone();
            idct_add(&coefficients, &mut plane, offset, 16);
            plane
        });
        across_isas("inverse_walsh", || inverse_walsh(&coefficients));
    }
}

#[test]
fn inter_prediction_matches_scalar() {
    let mut random = Random(0x6a9);
    for round in 0..96 {
        let reference = if round == 0 {
            Plane {
                data: vec![255; 64 * 48],
                width: 64,
                height: 48,
            }
        } else {
            random.plane(64, 48)
        };
        let filters = if round % 2 == 0 {
            &SIXTAP_FILTERS
        } else {
            &BILINEAR_FILTERS
        };
        for size in [4usize, 8, 16] {
            let x = random.below((64 / size) as u32) as usize * size;
            let y = random.below((48 / size) as u32) as usize * size;
            // Mostly inside the reference, sometimes well beyond its edges.
            let reach = if round % 5 == 0 { 400 } else { 60 };
            let mv_x = random.below(2 * reach + 1) as i32 - reach as i32;
            let mv_y = random.below(2 * reach + 1) as i32 - reach as i32;
            across_isas("predict_inter", || {
                let mut output = Plane::new(64, 48);
                predict_inter(
                    &reference,
                    &mut output,
                    x,
                    y,
                    size,
                    size,
                    mv_x,
                    mv_y,
                    filters,
                );
                output.data
            });
        }
    }
}

#[test]
fn tm_prediction_matches_scalar() {
    let mut random = Random(0x7a);
    for round in 0..128 {
        let mut above = [0u8; 21];
        let mut left = [0u8; 16];
        for sample in above.iter_mut().chain(&mut left) {
            *sample = match round % 3 {
                0 => random.sample(),
                1 => 255,
                _ => 0,
            };
        }
        if round % 3 != 0 {
            // Saturate the other way at the corner.
            above[0] = 255 - above[0];
        }
        let luma = Edges::<16, 21> { above, left };
        let mut chroma_above = [0u8; 9];
        chroma_above.copy_from_slice(&above[..9]);
        let mut chroma_left = [0u8; 8];
        chroma_left.copy_from_slice(&left[..8]);
        let chroma = Edges::<8, 9> {
            above: chroma_above,
            left: chroma_left,
        };
        across_isas("tm 16x16", || {
            let mut plane = vec![0u8; 32 * 32];
            predict_block(TM_PRED, &luma, true, true, &mut plane, 33, 32);
            plane
        });
        across_isas("tm 8x8", || {
            let mut plane = vec![0u8; 16 * 16];
            predict_block(TM_PRED, &chroma, true, true, &mut plane, 17, 16);
            plane
        });
    }
}

#[test]
fn loop_filter_matches_scalar() {
    let mut random = Random(0x100f);
    let (mb_cols, mb_rows) = (4, 3);
    for round in 0..96 {
        let planes = if round % 8 == 7 {
            [
                random.plane(64, 48),
                random.plane(32, 24),
                random.plane(32, 24),
            ]
        } else {
            [
                random.smooth_plane(64, 48),
                random.smooth_plane(32, 24),
                random.smooth_plane(32, 24),
            ]
        };
        let macroblocks: Vec<MacroblockFilter> = (0..mb_cols * mb_rows)
            .map(|_| MacroblockFilter {
                level: match random.below(4) {
                    0 => 63,
                    1 => 0,
                    _ => random.below(64) as u8,
                },
                inner_edges: random.below(2) == 0,
            })
            .collect();
        let frame = FrameFilter {
            simple: round % 4 == 1,
            sharpness: random.below(8) as u8,
            key_frame: round % 3 == 0,
        };
        across_isas("filter_frame", || {
            let mut planes = planes.clone();
            filter_frame(&mut planes, &macroblocks, mb_cols, frame);
            planes.map(|plane| plane.data)
        });
    }
}
