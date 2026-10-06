//! Per-stage inputs for the Vorbis groups in `benches/audio_decode.rs`.
//!
//! Internal and unstable. Each `run_*` drives one synthesis kernel over
//! [`BLOCKS`] long blocks of deterministic content on the instruction set
//! [`crate::simd`] currently selects, and returns an eight-byte fold over every
//! sample it produced. The fold keeps the output out of the timed loop's
//! allocations while still letting the benchmark's bit-exactness guard catch an
//! arm that diverged anywhere.

use super::Imdct;

/// Spectral coefficients in a long block: libvorbis's 2048-sample long block,
/// the one nearly every packet of a 44.1 or 48 kHz stream uses.
pub const LONG_SPECTRUM: usize = 1024;

/// Blocks one `run_*` call covers.
pub const BLOCKS: usize = 64;

/// Precomputed inputs and output buffers for every stage group.
pub struct VorbisStageInputs {
    imdct: Imdct,
    spectra: Vec<f32>,
    imdct_out: Vec<f32>,
    window: Vec<f32>,
    window_rev: Vec<f32>,
    previous: Vec<f32>,
    current: Vec<f32>,
    synthesized: Vec<f32>,
    magnitude: Vec<f32>,
    angle: Vec<f32>,
    coupled_magnitude: Vec<f32>,
    coupled_angle: Vec<f32>,
    floor: Vec<f32>,
    residue: Vec<f32>,
    product: Vec<f32>,
}

impl Default for VorbisStageInputs {
    fn default() -> Self {
        Self::new()
    }
}

impl VorbisStageInputs {
    #[must_use]
    pub fn new() -> Self {
        let samples = BLOCKS * LONG_SPECTRUM;
        let mut random = Random(0x5eed_0572);
        // A spectrum that decays with frequency, as music's does, around
        // the unit scale the decoder's floor-times-residue product has.
        let spectra = (0..samples)
            .map(|i| {
                let bin = (i % LONG_SPECTRUM) as f32;
                random.signed() / (1.0 + bin / 64.0)
            })
            .collect();
        let window = long_window(2 * LONG_SPECTRUM);
        let window_rev = window.iter().rev().copied().collect();
        // Slightly over unit scale, so the clamp has samples to clamp.
        let previous = (0..samples).map(|_| 1.2 * random.signed()).collect();
        let current = (0..samples).map(|_| 1.2 * random.signed()).collect();
        let magnitude = (0..samples).map(|_| random.signed()).collect();
        let angle = (0..samples).map(|_| random.signed()).collect();
        let floor = (0..samples).map(|_| random.unit() * 0.1).collect();
        let residue = (0..samples).map(|_| random.signed() * 8.0).collect();
        VorbisStageInputs {
            imdct: Imdct::new(LONG_SPECTRUM),
            spectra,
            imdct_out: vec![0.0; 2 * samples],
            window,
            window_rev,
            previous,
            current,
            synthesized: vec![0.0; samples],
            magnitude,
            angle,
            coupled_magnitude: vec![0.0; samples],
            coupled_angle: vec![0.0; samples],
            floor,
            residue,
            product: vec![0.0; samples],
        }
    }

    /// Output samples per channel one `run_*` call stands for: a long block
    /// contributes half its length.
    #[must_use]
    pub fn samples_per_run(&self) -> u64 {
        (BLOCKS * LONG_SPECTRUM) as u64
    }

    /// The inverse MDCT of [`BLOCKS`] long-block spectra.
    pub fn run_imdct(&mut self) -> u64 {
        for (spectrum, out) in self
            .spectra
            .chunks_exact(LONG_SPECTRUM)
            .zip(self.imdct_out.chunks_exact_mut(2 * LONG_SPECTRUM))
        {
            self.imdct.imdct(spectrum, out);
        }
        fold(&self.imdct_out)
    }

    /// Windowed overlap-add of [`BLOCKS`] long-block halves, then the output
    /// clamp, as the decoder runs them back to back.
    pub fn run_overlap_add(&mut self) -> u64 {
        for ((out, left), right) in self
            .synthesized
            .chunks_exact_mut(LONG_SPECTRUM)
            .zip(self.previous.chunks_exact(LONG_SPECTRUM))
            .zip(self.current.chunks_exact(LONG_SPECTRUM))
        {
            super::overlap_add(out, left, right, &self.window, &self.window_rev);
            super::clamp_unit(out);
        }
        fold(&self.synthesized)
    }

    /// Inverse coupling of [`BLOCKS`] channel pairs. The kernel works in
    /// place, so each run starts from a fresh copy of the coupled residues.
    pub fn run_coupling(&mut self) -> u64 {
        self.coupled_magnitude.copy_from_slice(&self.magnitude);
        self.coupled_angle.copy_from_slice(&self.angle);
        for (magnitude, angle) in self
            .coupled_magnitude
            .chunks_exact_mut(LONG_SPECTRUM)
            .zip(self.coupled_angle.chunks_exact_mut(LONG_SPECTRUM))
        {
            super::inverse_coupling(magnitude, angle);
        }
        fold(&self.coupled_magnitude) ^ fold(&self.coupled_angle).rotate_left(1)
    }

    /// The floor-times-residue product over [`BLOCKS`] channels, from a fresh
    /// copy of the floor each run for the same reason.
    pub fn run_floor_product(&mut self) -> u64 {
        self.product.copy_from_slice(&self.floor);
        for (floor, residue) in self
            .product
            .chunks_exact_mut(LONG_SPECTRUM)
            .zip(self.residue.chunks_exact(LONG_SPECTRUM))
        {
            super::apply_floor(floor, residue);
        }
        fold(&self.product)
    }
}

/// An order-sensitive fold of every sample's bits. Each sample is weighted by
/// an odd multiplier, which is invertible modulo 2^64, so changing any one
/// sample always changes the result.
#[must_use]
pub fn fold(samples: &[f32]) -> u64 {
    samples.iter().enumerate().fold(0_u64, |acc, (i, &s)| {
        acc.wrapping_add(u64::from(s.to_bits()).wrapping_mul(2 * i as u64 + 1))
    })
}

/// The rising half of the Vorbis window for a `size`-sample block (Vorbis I
/// section 4.3.1).
fn long_window(size: usize) -> Vec<f32> {
    let half = size / 2;
    (0..half)
        .map(|i| {
            let x = (i as f64 + 0.5) / half as f64 * std::f64::consts::FRAC_PI_2;
            (std::f64::consts::FRAC_PI_2 * x.sin().powi(2)).sin() as f32
        })
        .collect()
}

/// A small deterministic generator, so every process times the same content.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `0.0..1.0`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1_u64 << 24) as f32
    }

    /// Uniform in `-1.0..1.0`.
    fn signed(&mut self) -> f32 {
        2.0 * self.unit() - 1.0
    }
}
