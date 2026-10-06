//! The Vorbis inverse MDCT, owned by this crate so its vector arms can be
//! dispatched through [`crate::simd`] and held bit-exact with its scalar
//! reference.
//!
//! It replaces `symphonia_core::dsp::mdct::Imdct`. Symphonia's only vector
//! path is `rustfft`'s behind its `opt-simd` feature, which picks an
//! instruction set on its own - out of reach of [`crate::simd::set_override`]
//! - and rounds differently from its scalar FFT, which would break the
//! crate-wide promise that the override never changes output (issue #572).
//! The transform here keeps Symphonia's pre- and post-twiddle arithmetic and
//! its FFT's twiddle factors, so it is the same IMDCT up to rounding.

// Symphonia
// Copyright (c) 2019-2022 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The twiddle tables follow `symphonia-core` 0.5.5's IMDCT and FFT.

use crate::simd::SimdIsa;

use super::kernels::{self, Scratch};

/// The constant tables for one transform size.
pub(crate) struct Plan {
    /// `cos(pi / n * (k + 1/8))` for `k` in `0..n/2`.
    pub(crate) tw_re: Box<[f32]>,
    /// `sin(pi / n * (k + 1/8))`.
    pub(crate) tw_im: Box<[f32]>,
    /// `-sin(pi / n * (k + 1/8))`, so the pre-twiddle needs no sign flip.
    pub(crate) tw_neg_im: Box<[f32]>,
    /// The `n/2`-point bit-reversal permutation.
    pub(crate) perm: Box<[u32]>,
    /// `cos(pi j / half)` for every FFT stage, the stage of half-length
    /// `half` occupying `half - 1 .. 2 * half - 1`.
    stage_re: Box<[f32]>,
    /// `-sin(pi j / half)`, laid out as `stage_re`.
    stage_im: Box<[f32]>,
}

impl Plan {
    fn new(n: usize) -> Self {
        let n2 = n / 2;
        let pi_n = std::f64::consts::PI / n as f64;
        let mut tw_re = Vec::with_capacity(n2);
        let mut tw_im = Vec::with_capacity(n2);
        for k in 0..n2 {
            let theta = pi_n * (0.125 + k as f64);
            tw_re.push(theta.cos() as f32);
            tw_im.push(theta.sin() as f32);
        }
        let tw_neg_im = tw_im.iter().map(|&v| -v).collect();

        let bits = n2.trailing_zeros();
        let perm = (0..n2 as u32)
            .map(|i| i.reverse_bits() >> (u32::BITS - bits))
            .collect();

        let mut stage_re = Vec::with_capacity(n2);
        let mut stage_im = Vec::with_capacity(n2);
        let mut half = 1;
        while half < n2 {
            let theta = std::f64::consts::PI / half as f64;
            for j in 0..half {
                let angle = theta * j as f64;
                stage_re.push(angle.cos() as f32);
                stage_im.push(-angle.sin() as f32);
            }
            half *= 2;
        }

        Plan {
            tw_re: tw_re.into_boxed_slice(),
            tw_im: tw_im.into_boxed_slice(),
            tw_neg_im,
            perm,
            stage_re: stage_re.into_boxed_slice(),
            stage_im: stage_im.into_boxed_slice(),
        }
    }

    /// The twiddles of the FFT stage with half-length `half`.
    #[inline(always)]
    pub(crate) fn stage_twiddles(&self, half: usize) -> (&[f32], &[f32]) {
        let range = half - 1..2 * half - 1;
        (&self.stage_re[range.clone()], &self.stage_im[range])
    }
}

/// An inverse MDCT of an `n`-point spectrum into `2n` samples, unscaled.
pub(crate) struct Imdct {
    plan: Plan,
    folded_re: Box<[f32]>,
    folded_im: Box<[f32]>,
    re: Box<[f32]>,
    im: Box<[f32]>,
}

impl Imdct {
    /// A transform of `n` spectral coefficients. `n` must be a power of two
    /// and at least 8; Vorbis's smallest block, 64 samples, has `n = 32`.
    pub(crate) fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 8,
            "IMDCT size {n} is unsupported"
        );
        let buffer = || vec![0.0; n / 2].into_boxed_slice();
        Imdct {
            plan: Plan::new(n),
            folded_re: buffer(),
            folded_im: buffer(),
            re: buffer(),
            im: buffer(),
        }
    }

    /// Transforms `spec` (`n` long) into `out` (`2n` long) on the active
    /// instruction set.
    pub(crate) fn imdct(&mut self, spec: &[f32], out: &mut [f32]) {
        self.imdct_with(super::active_isa(), spec, out);
    }

    /// [`Imdct::imdct`] on `isa`, which must be one this host can execute.
    pub(crate) fn imdct_with(&mut self, isa: SimdIsa, spec: &[f32], out: &mut [f32]) {
        let n = self.re.len() * 2;
        assert_eq!(spec.len(), n);
        assert_eq!(out.len(), 2 * n);
        let plan = &self.plan;
        let scratch = Scratch {
            folded_re: &mut self.folded_re,
            folded_im: &mut self.folded_im,
            re: &mut self.re,
            im: &mut self.im,
        };
        match isa {
            // SAFETY: the caller only names an instruction set this host
            // supports (see `super::active_isa`).
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => unsafe { super::imdct_avx2(plan, spec, out, scratch) },
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => unsafe { super::imdct_sse41(plan, spec, out, scratch) },
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => unsafe { super::imdct_neon(plan, spec, out, scratch) },
            _ => kernels::imdct_scalar(plan, spec, out, scratch),
        }
    }
}
