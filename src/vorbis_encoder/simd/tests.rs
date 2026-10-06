//! Every vector kernel against its scalar reference, on random input and on
//! the edge cases that separate "close" from "bit-exact": signed zeros,
//! infinities, NaN, denormals, values on a rounding boundary of the integer
//! conversions, and lengths that leave a scalar tail. Then whole encodes,
//! which must be byte-identical under every instruction set.

use super::*;
use crate::simd::{self, SimdIsa};
use crate::vorbis_encoder::VorbisEncoder;
use crate::vorbis_encoder::smallft::DrftLookup;

/// The vector instruction sets this host runs; empty on a host without one
/// (and on targets with no vector kernels), where every test below is then
/// only a check of the scalar arm against itself.
fn vector_isas() -> Vec<SimdIsa> {
    simd::available()
        .into_iter()
        .filter(|&isa| isa != SimdIsa::Scalar)
        .collect()
}

/// xorshift64*: deterministic, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
    /// Audio-like: mostly moderate values, with signed zeros, denormals,
    /// huge values and non-finite values mixed in when `specials` is set.
    fn sample(&mut self, specials: bool) -> f32 {
        if specials {
            match self.below(24) {
                0 => return 0.0,
                1 => return -0.0,
                2 => return f32::from_bits(1 + self.below(1 << 23) as u32),
                3 => return -f32::MIN_POSITIVE / 3.0,
                4 => return f32::INFINITY,
                5 => return f32::NEG_INFINITY,
                6 => return f32::NAN,
                7 => return 3.0e38,
                8 => return -1.0e30,
                _ => {}
            }
        }
        self.range(-1.0, 1.0) * [1.0, 1e-3, 30.0, 1e4][self.below(4)]
    }
    fn samples(&mut self, len: usize, specials: bool) -> Vec<f32> {
        (0..len).map(|_| self.sample(specials)).collect()
    }
}

/// Bit-for-bit equality, except that any NaN equals any NaN: the payload of
/// a NaN produced from NaN input is not something the scalar code pins down
/// either (LLVM may commute the operands of a scalar add or multiply).
fn assert_same(label: &str, isa: SimdIsa, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label} ({})", isa.name());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
        assert!(
            same,
            "{label} ({}): element {i} is {g:e} ({:#010x}), scalar gives {w:e} ({:#010x})",
            isa.name(),
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn mdct_matches_scalar_at_every_block_size() {
    let mut rng = Rng::new(1);
    for log2n in 6..=13 {
        let n = 1usize << log2n;
        let m = MdctLookup::new(n);
        for specials in [false, true] {
            let input = rng.samples(n, specials);
            let mut want = vec![0f32; n / 2];
            let mut w = vec![0f32; n];
            mdct_forward(SimdIsa::Scalar, &m, &input, &mut want, &mut w);
            for isa in vector_isas() {
                let mut got = vec![0f32; n / 2];
                let mut w = vec![f32::NAN; n];
                mdct_forward(isa, &m, &input, &mut got, &mut w);
                assert_same(&format!("mdct n={n} specials={specials}"), isa, &got, &want);
            }
        }
    }
}

#[test]
fn fft_matches_scalar_at_every_block_size() {
    let mut rng = Rng::new(2);
    for log2n in 2..=13 {
        let n = 1usize << log2n;
        let look = DrftLookup::new(n);
        for specials in [false, true] {
            let input = rng.samples(n, specials);
            let mut want = input.clone();
            look.forward(SimdIsa::Scalar, &mut want, &mut vec![0f32; n]);
            for isa in vector_isas() {
                let mut got = input.clone();
                look.forward(isa, &mut got, &mut vec![f32::NAN; n]);
                assert_same(&format!("fft n={n} specials={specials}"), isa, &got, &want);
            }
        }
    }
}

/// The FFT passes on shapes the power-of-two transforms never produce, so
/// every vector loop also runs with a ragged scalar tail.
#[test]
fn fft_passes_match_scalar_on_ragged_shapes() {
    let mut rng = Rng::new(3);
    for l1 in 1..=9 {
        let cc = rng.samples(4 * l1, true);
        let mut want = vec![0f32; 4 * l1];
        dradf4_ido1(SimdIsa::Scalar, l1, &cc, &mut want);
        for isa in vector_isas() {
            let mut got = vec![f32::NAN; 4 * l1];
            dradf4_ido1(isa, l1, &cc, &mut got);
            assert_same(&format!("dradf4 ido=1 l1={l1}"), isa, &got, &want);
        }
    }
    for ido in [3usize, 4, 5, 6, 8, 10, 12, 16, 18, 64] {
        for l1 in [1usize, 2, 3] {
            let t0 = l1 * ido;
            let cc = rng.samples(4 * t0, true);
            let wa: Vec<Vec<f32>> = (0..3).map(|_| rng.samples(ido, false)).collect();
            let init = rng.samples(4 * t0, false);
            let mut want = init.clone();
            dradf4_twiddle(
                SimdIsa::Scalar,
                ido,
                l1,
                &cc,
                &mut want,
                &wa[0],
                &wa[1],
                &wa[2],
            );
            let mut want2 = init[..2 * t0].to_vec();
            dradf2_twiddle(SimdIsa::Scalar, ido, l1, &cc[..2 * t0], &mut want2, &wa[0]);
            for isa in vector_isas() {
                let mut got = init.clone();
                dradf4_twiddle(isa, ido, l1, &cc, &mut got, &wa[0], &wa[1], &wa[2]);
                assert_same(&format!("dradf4 ido={ido} l1={l1}"), isa, &got, &want);
                let mut got2 = init[..2 * t0].to_vec();
                dradf2_twiddle(isa, ido, l1, &cc[..2 * t0], &mut got2, &wa[0]);
                assert_same(&format!("dradf2 ido={ido} l1={l1}"), isa, &got2, &want2);
            }
        }
    }
}

/// Monotonic running sums of positive weights, like the ones
/// `bark_noise_hybridmp` builds, with an optional sprinkling of specials.
fn noise_sums(rng: &mut Rng, n: usize, specials: bool) -> NoiseSums {
    let mut column = |scale: f32| {
        let mut acc = 0f32;
        (0..n)
            .map(|_| {
                acc += rng.range(0.0, scale);
                if specials && rng.below(16) == 0 {
                    rng.sample(true)
                } else {
                    acc
                }
            })
            .collect::<Vec<f32>>()
    };
    NoiseSums {
        n: column(4e4),
        x: column(4e6),
        xx: column(4e9),
        y: column(4e6),
        xy: column(4e8),
    }
}

#[test]
fn noise_fits_match_scalar() {
    let mut rng = Rng::new(4);
    for n in [1usize, 3, 4, 5, 64, 129, 1024] {
        for specials in [false, true] {
            let sums = noise_sums(&mut rng, n, specials);
            let noise_in = rng.samples(n, specials);
            let offset = [0.0, 140.0, -3.5][rng.below(3)];

            // Bark windows: random (lo, hi) inside the spectrum, mirrored
            // for the reflected half.
            let windows: Vec<i32> = (0..n)
                .map(|_| {
                    let lo = rng.below(n) as i32;
                    let hi = rng.below(n) as i32;
                    (lo << 16) | hi
                })
                .collect();
            let mirrored: Vec<i32> = windows
                .iter()
                .map(|&w| (-(w >> 16).max(1) << 16) | (w & 0xffff))
                .collect();
            let start = rng.below(n.min(5) + 1).min(n);
            // A mirrored window needs a lower edge in 1..n.
            let shapes: &[(&Vec<i32>, bool)] = if n > 1 {
                &[(&windows, false), (&mirrored, true)]
            } else {
                &[(&windows, false)]
            };
            for &(b, reflect) in shapes {
                let mut want = noise_in.clone();
                noise_bark(
                    SimdIsa::Scalar,
                    &sums,
                    b,
                    start,
                    n,
                    reflect,
                    offset,
                    &mut want,
                );
                for isa in vector_isas() {
                    let mut got = noise_in.clone();
                    noise_bark(isa, &sums, b, start, n, reflect, offset, &mut got);
                    let label = format!("noise_bark n={n} reflect={reflect} specials={specials}");
                    assert_same(&label, isa, &got, &want);
                }
            }

            // Fixed windows: the same segmentation `bark_noise_hybridmp` uses.
            for fixed in [1i32, 2, 7, 8, 9, 31] {
                let ni = n as i32;
                let edges = |i: usize| (i as i32 + fixed / 2, i as i32 + fixed / 2 - fixed);
                // `bark_noise_hybridmp`'s scan, plus the bound on the
                // mirrored edge that its block sizes always satisfy.
                let reflected = (0..n)
                    .find(|&i| edges(i).0 >= ni || edges(i).1 >= 0 || -edges(i).1 >= ni)
                    .unwrap_or(n);
                let fitted = (reflected..n)
                    .find(|&i| edges(i).0 >= ni || edges(i).1 < 0)
                    .unwrap_or(n);
                for (s, e, reflect) in [(0, reflected, true), (reflected, fitted, false)] {
                    let mut want = noise_in.clone();
                    noise_fixed(
                        SimdIsa::Scalar,
                        &sums,
                        fixed,
                        s,
                        e,
                        reflect,
                        offset,
                        &mut want,
                    );
                    for isa in vector_isas() {
                        let mut got = noise_in.clone();
                        noise_fixed(isa, &sums, fixed, s, e, reflect, offset, &mut got);
                        let label = format!(
                            "noise_fixed n={n} fixed={fixed} reflect={reflect} specials={specials}"
                        );
                        assert_same(&label, isa, &got, &want);
                    }
                }
            }

            let fit = (
                rng.sample(specials),
                rng.sample(specials),
                rng.sample(specials),
            );
            for lower_only in [false, true] {
                let mut want = noise_in.clone();
                noise_extrapolate(
                    SimdIsa::Scalar,
                    fit,
                    start,
                    n,
                    offset,
                    lower_only,
                    &mut want,
                );
                for isa in vector_isas() {
                    let mut got = noise_in.clone();
                    noise_extrapolate(isa, fit, start, n, offset, lower_only, &mut got);
                    let label = format!("noise_extrapolate n={n} lower_only={lower_only}");
                    assert_same(&label, isa, &got, &want);
                }
            }
        }
    }
}

#[test]
fn noise_compand_matches_scalar_on_every_index_boundary() {
    let mut rng = Rng::new(5);
    let compand: Vec<f32> = (0..40).map(|_| rng.range(-10.0, 10.0)).collect();
    // `int dB = logmask + .5` around every boundary of the 40-entry table,
    // where the f64 addition (not an f32 one) decides the index.
    let mut logmask: Vec<f32> = Vec::new();
    for k in -2..=42 {
        let edge = k as f32 - 0.5;
        logmask.extend([
            edge,
            f32::from_bits(edge.to_bits() + 1),
            f32::from_bits(edge.to_bits().wrapping_sub(1)),
            k as f32,
            0.499_999_97,
            -0.0,
        ]);
    }
    logmask.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e9, -3e9, 1e38]);
    logmask.extend(rng.samples(101, true));
    let n = logmask.len();
    let logmdct = rng.samples(n, true);
    let work = rng.samples(n, true);
    let mut want = logmask.clone();
    noise_compand(SimdIsa::Scalar, &logmdct, &work, &mut want, &compand);
    for isa in vector_isas() {
        let mut got = logmask.clone();
        noise_compand(isa, &logmdct, &work, &mut got, &compand);
        assert_same("noise_compand", isa, &got, &want);
    }
}

#[test]
fn floor_accumulation_matches_scalar() {
    let mut rng = Rng::new(6);
    let n = 1024usize;
    // Masks around `vorbis_dBquant`'s truncation boundaries and its clamps.
    let mut flr: Vec<f32> = (0..n)
        .map(|i| match i % 8 {
            0 => ((rng.below(1100) as f32) - 1023.5) / 7.314_285_8,
            1 => rng.sample(true),
            2 => -140.0 + rng.range(-1.0, 1.0),
            _ => rng.range(-160.0, 10.0),
        })
        .collect();
    flr[17] = -1023.5 / 7.314_285_8;
    flr[18] = f32::NAN;
    flr[19] = f32::INFINITY;
    let mdct: Vec<f32> = flr
        .iter()
        .map(|&f| match rng.below(4) {
            0 => f,
            1 => f - 1.0,
            2 => rng.sample(true),
            _ => f + rng.range(-20.0, 20.0),
        })
        .collect();
    for _ in 0..400 {
        let x0 = rng.below(n) as i32;
        let x1 = (x0 + rng.below(70) as i32 - 2).min(n as i32 - 1);
        let att = [0.0, -6.0, 3.25][rng.below(3)];
        let want = accumulate_fit(SimdIsa::Scalar, &flr, &mdct, x0, x1, att);
        for isa in vector_isas() {
            let got = accumulate_fit(isa, &flr, &mdct, x0, x1, att);
            assert_eq!(got, want, "accumulate_fit {x0}..={x1} ({})", isa.name());
        }
    }
}

#[test]
fn log_spectra_match_scalar() {
    let mut rng = Rng::new(7);
    for n in [4usize, 6, 8, 10, 18, 64, 256, 2048] {
        for specials in [false, true] {
            let mdct = rng.samples(n / 2, specials);
            let mut want = vec![0f32; n / 2];
            log_mdct(SimdIsa::Scalar, &mdct, &mut want);

            let spectrum = rng.samples(n, specials);
            let scale_db = rng.range(-80.0, 0.0);
            let mut want_fft = spectrum.clone();
            log_fft(SimdIsa::Scalar, &mut want_fft, n, scale_db);
            for isa in vector_isas() {
                let mut got = vec![f32::NAN; n / 2];
                log_mdct(isa, &mdct, &mut got);
                assert_same(&format!("log_mdct n={n}"), isa, &got, &want);

                let mut got_fft = spectrum.clone();
                log_fft(isa, &mut got_fft, n, scale_db);
                assert_same(&format!("log_fft n={n}"), isa, &got_fft, &want_fft);
            }
        }
    }
}

/// Interleaved test signal: tones, a noise floor, transients (to force short
/// blocks), a silent stretch, a clipped stretch and a denormal tail.
fn test_signal(rate: u32, channels: u16, seconds: f32) -> Vec<f32> {
    let frames = (rate as f32 * seconds) as usize;
    let mut rng = Rng::new(u64::from(rate) * 7 + u64::from(channels));
    let mut pcm = Vec::with_capacity(frames * usize::from(channels));
    for i in 0..frames {
        let t = i as f32 / rate as f32;
        for c in 0..channels {
            let tone = 0.3 * (std::f32::consts::TAU * (220.0 + 110.0 * f32::from(c)) * t).sin()
                + 0.1 * (std::f32::consts::TAU * 3150.0 * t).sin();
            let burst = if i % (rate as usize / 3) < 200 {
                0.9
            } else {
                0.0
            };
            let noise = rng.range(-0.05, 0.05);
            let s = match (i * 8) / frames {
                2 => 0.0,
                5 => (4.0 * tone).clamp(-1.0, 1.0),
                7 => tone * 1e-39,
                _ => tone + burst * rng.range(-1.0, 1.0) + noise,
            };
            pcm.push(s);
        }
    }
    pcm
}

fn encode(rate: u32, channels: u16, quality: f32, pcm: &[f32]) -> Vec<(Vec<u8>, u64)> {
    let mut encoder = VorbisEncoder::new(rate, channels, quality).expect("encoder");
    let mut packets = Vec::new();
    for chunk in pcm.chunks(1500 * usize::from(channels)) {
        packets.extend(encoder.encode(chunk).expect("encode"));
    }
    packets.extend(encoder.finish().expect("finish"));
    packets
}

/// The acceptance criterion of issue #573: encoded output is byte-identical
/// with SIMD on and off.
#[test]
fn encodes_are_byte_identical_under_every_instruction_set() {
    let configs: [(u32, u16, f32); 6] = [
        (44_100, 2, 0.3),
        (48_000, 2, 0.6),
        (44_100, 1, 0.0),
        (22_050, 2, -0.1),
        (8_000, 1, 0.5),
        (96_000, 2, 1.0),
    ];
    let _guard = simd::test_lock();
    for (rate, channels, quality) in configs {
        let pcm = test_signal(rate, channels, 1.5);
        simd::set_override(Some(SimdIsa::Scalar));
        assert_eq!(active_isa(), SimdIsa::Scalar);
        let want = encode(rate, channels, quality, &pcm);
        for isa in vector_isas() {
            simd::set_override(Some(isa));
            assert_eq!(active_isa(), isa);
            let got = encode(rate, channels, quality, &pcm);
            assert_eq!(
                got.len(),
                want.len(),
                "{rate} Hz x{channels} q{quality} ({})",
                isa.name()
            );
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g == w,
                    "{rate} Hz x{channels} q{quality}: packet {k} differs under {}",
                    isa.name()
                );
            }
        }
    }
    simd::set_override(None);
}
