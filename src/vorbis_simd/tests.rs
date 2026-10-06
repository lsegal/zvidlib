//! Every vector arm against its scalar reference, bit for bit, and the
//! scalar inverse MDCT against the transform's definition.

use super::*;
use crate::simd::{self, SimdIsa};

/// Every instruction set this host can run other than the scalar reference.
fn vector_isas() -> Vec<SimdIsa> {
    simd::available()
        .into_iter()
        .filter(|&isa| isa != SimdIsa::Scalar)
        .collect()
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1_u64 << 23) as f32 - 1.0
    }

    /// Random values with an edge case mixed in roughly one time in four:
    /// signed zeros, the clamp and coupling boundaries, subnormals, and very
    /// large magnitudes.
    fn mixed(&mut self, len: usize) -> Vec<f32> {
        const EDGES: [f32; 12] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0e-40,
            -1.0e-40,
            f32::MIN_POSITIVE,
            3.0e30,
            -3.0e30,
            1.000_000_1,
            -1.000_000_1,
            0.5,
        ];
        (0..len)
            .map(|_| {
                let pick = self.next();
                if pick % 4 == 0 {
                    EDGES[(pick >> 8) as usize % EDGES.len()]
                } else {
                    self.signed() * 2.0
                }
            })
            .collect()
    }
}

fn assert_bits_eq(label: &str, isa: SimdIsa, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: {} differs from scalar at {i}: {a:e} vs {e:e}",
            isa.name()
        );
    }
}

/// Lengths that cover an empty slice, partial vectors on both sides of each
/// lane width, and a decoder-sized run.
const LENGTHS: [usize; 12] = [0, 1, 3, 4, 5, 7, 8, 9, 15, 17, 33, 1024];

#[test]
fn imdct_vector_arms_match_scalar_bit_for_bit() {
    let mut random = Random(0x1234_5678_9abc_def1);
    for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let inputs = [
            (0..n).map(|_| random.signed()).collect::<Vec<_>>(),
            random.mixed(n),
            vec![0.0; n],
            (0..n)
                .map(|i| if i == n / 3 { 1.0 } else { -0.0 })
                .collect(),
            (0..n)
                .map(|i| if i % 2 == 0 { 1.0e20 } else { -1.0e-20 })
                .collect(),
        ];
        for spec in &inputs {
            let mut expected = vec![0.0; 2 * n];
            Imdct::new(n).imdct_with(SimdIsa::Scalar, spec, &mut expected);
            for isa in vector_isas() {
                let mut actual = vec![f32::NAN; 2 * n];
                Imdct::new(n).imdct_with(isa, spec, &mut actual);
                assert_bits_eq(&format!("imdct n={n}"), isa, &actual, &expected);
            }
        }
    }
}

/// The scalar reference against the IMDCT's definition (Vorbis I section
/// 4.3.7 uses Symphonia's unscaled convention), evaluated in `f64`.
#[test]
fn scalar_imdct_matches_the_definition() {
    let mut random = Random(0x0bad_cafe_f00d_d00d);
    for n in [8, 32, 256, 1024] {
        let spec: Vec<f32> = (0..n).map(|_| random.signed()).collect();
        let mut actual = vec![0.0; 2 * n];
        Imdct::new(n).imdct_with(SimdIsa::Scalar, &spec, &mut actual);
        let pi_2n = std::f64::consts::PI / (4 * n) as f64;
        for (i, &value) in actual.iter().enumerate() {
            let expected: f64 = spec
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    f64::from(x) * (pi_2n * ((2 * i + 1 + n) * (2 * j + 1)) as f64).cos()
                })
                .sum();
            // Rounding grows with the transform's length and its sums.
            let tolerance = 2.0e-6 * n as f64;
            assert!(
                (f64::from(value) - expected).abs() <= tolerance,
                "n={n} sample {i}: {value} vs {expected}"
            );
        }
    }
}

/// The transform it replaced, to the rounding the two FFT layouts differ by.
#[test]
fn scalar_imdct_matches_symphonias() {
    let mut random = Random(0x5eed_5eed_5eed_5eed);
    for n in [32, 128, 1024, 4096] {
        let spec: Vec<f32> = (0..n).map(|_| random.signed()).collect();
        let mut ours = vec![0.0; 2 * n];
        let mut theirs = vec![0.0; 2 * n];
        Imdct::new(n).imdct_with(SimdIsa::Scalar, &spec, &mut ours);
        symphonia_core::dsp::mdct::Imdct::new(n).imdct(&spec, &mut theirs);
        let peak = theirs.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert!(
                (a - b).abs() <= 1.0e-5 * peak,
                "n={n} sample {i}: {a} vs Symphonia's {b}"
            );
        }
    }
}

#[test]
fn overlap_add_vector_arms_match_scalar_bit_for_bit() {
    let mut random = Random(0x0f0f_1e1e_2d2d_3c3c);
    for len in LENGTHS {
        let left = random.mixed(len);
        let right = random.mixed(len);
        let win: Vec<f32> = (0..len).map(|_| random.signed().abs()).collect();
        let win_rev: Vec<f32> = win.iter().rev().copied().collect();
        let mut expected = vec![0.0; len];
        overlap_add_with(
            SimdIsa::Scalar,
            &mut expected,
            &left,
            &right,
            &win,
            &win_rev,
        );
        for isa in vector_isas() {
            let mut actual = vec![f32::NAN; len];
            overlap_add_with(isa, &mut actual, &left, &right, &win, &win_rev);
            assert_bits_eq(&format!("overlap_add len={len}"), isa, &actual, &expected);
        }
    }
}

#[test]
fn clamp_vector_arms_match_scalar_bit_for_bit() {
    let mut random = Random(0x7777_1111_3333_5555);
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.000_000_1,
        -1.000_000_1,
        0.999_999_94,
        -0.999_999_94,
        1.0e-40,
        f32::MAX,
        f32::MIN,
    ];
    for len in LENGTHS {
        let mut input = random.mixed(len);
        for (slot, &special) in input.iter_mut().zip(specials.iter().cycle()).step_by(3) {
            *slot = special;
        }
        let mut expected = input.clone();
        clamp_unit_with(SimdIsa::Scalar, &mut expected);
        for isa in vector_isas() {
            let mut actual = input.clone();
            clamp_unit_with(isa, &mut actual);
            assert_bits_eq(&format!("clamp len={len}"), isa, &actual, &expected);
        }
    }
}

#[test]
fn clamp_is_f32_clamp() {
    let input = [2.0, -2.0, 0.25, -0.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut output = input;
    clamp_unit_with(SimdIsa::Scalar, &mut output);
    for (out, inp) in output.iter().zip(input) {
        assert_eq!(out.to_bits(), inp.clamp(-1.0, 1.0).to_bits());
    }
    let mut nan = [f32::NAN];
    clamp_unit_with(SimdIsa::Scalar, &mut nan);
    assert!(nan[0].is_nan());
}

#[test]
fn coupling_vector_arms_match_scalar_bit_for_bit() {
    let mut random = Random(0x2468_ace0_1357_9bdf);
    for len in LENGTHS {
        let magnitude = random.mixed(len);
        let angle = random.mixed(len);
        let mut expected = (magnitude.clone(), angle.clone());
        inverse_coupling_with(SimdIsa::Scalar, &mut expected.0, &mut expected.1);
        for isa in vector_isas() {
            let mut actual = (magnitude.clone(), angle.clone());
            inverse_coupling_with(isa, &mut actual.0, &mut actual.1);
            let label = format!("coupling len={len}");
            assert_bits_eq(&label, isa, &actual.0, &expected.0);
            assert_bits_eq(&label, isa, &actual.1, &expected.1);
        }
    }
}

/// All four sign quadrants, plus the zeros on their boundaries, against the
/// table in Vorbis I section 4.3.5.
#[test]
fn coupling_follows_the_specification() {
    let cases = [
        // (m, a) -> (new m, new a)
        ((3.0, 1.0), (3.0, 2.0)),
        ((3.0, -1.0), (2.0, 3.0)),
        ((-3.0, 1.0), (-3.0, -2.0)),
        ((-3.0, -1.0), (-2.0, -3.0)),
        ((0.0, 1.0), (0.0, 1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
    ];
    for isa in simd::available() {
        let mut magnitude: Vec<f32> = cases.iter().map(|((m, _), _)| *m).collect();
        let mut angle: Vec<f32> = cases.iter().map(|((_, a), _)| *a).collect();
        // Pad to a whole AVX2 vector so the vector arms take these cases.
        magnitude.resize(8, 0.0);
        angle.resize(8, 0.0);
        inverse_coupling_with(isa, &mut magnitude, &mut angle);
        for (i, (_, (m, a))) in cases.iter().enumerate() {
            assert_eq!(
                (magnitude[i], angle[i]),
                (*m, *a),
                "{} case {i}",
                isa.name()
            );
        }
    }
}

#[test]
fn floor_product_vector_arms_match_scalar_bit_for_bit() {
    let mut random = Random(0x1111_2222_3333_4444);
    for len in LENGTHS {
        let floor = random.mixed(len);
        let residue = random.mixed(len);
        let mut expected = floor.clone();
        apply_floor_with(SimdIsa::Scalar, &mut expected, &residue);
        for isa in vector_isas() {
            let mut actual = floor.clone();
            apply_floor_with(isa, &mut actual, &residue);
            assert_bits_eq(&format!("floor len={len}"), isa, &actual, &expected);
        }
    }
}

/// The public entry points follow the crate-wide override rather than a
/// detection of their own.
#[test]
fn dispatchers_follow_the_override() {
    let _guard = simd::test_lock();
    let mut random = Random(0x9999_8888_7777_6666);
    let spec: Vec<f32> = (0..256).map(|_| random.signed()).collect();
    let mut expected = vec![0.0; 512];
    Imdct::new(256).imdct_with(SimdIsa::Scalar, &spec, &mut expected);
    for isa in simd::available() {
        simd::set_override(Some(isa));
        assert_eq!(active_isa(), isa);
        let mut actual = vec![0.0; 512];
        Imdct::new(256).imdct(&spec, &mut actual);
        assert_bits_eq("imdct via override", isa, &actual, &expected);
    }
    simd::set_override(None);
    assert_eq!(active_isa(), simd::detected());
}

/// The benchmark surface's stages produce the same fold on every arm, which
/// is what the benchmark's own guard checks before it times anything.
#[test]
fn bench_stages_agree_across_instruction_sets() {
    let _guard = simd::test_lock();
    let mut inputs = bench::VorbisStageInputs::new();
    simd::set_override(Some(SimdIsa::Scalar));
    let expected = [
        inputs.run_imdct(),
        inputs.run_overlap_add(),
        inputs.run_coupling(),
        inputs.run_floor_product(),
    ];
    for isa in vector_isas() {
        simd::set_override(Some(isa));
        let actual = [
            inputs.run_imdct(),
            inputs.run_overlap_add(),
            inputs.run_coupling(),
            inputs.run_floor_product(),
        ];
        assert_eq!(actual, expected, "{}", isa.name());
    }
    simd::set_override(None);
}
