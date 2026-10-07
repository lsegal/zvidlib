// Symphonia
// Copyright (c) 2019-2022 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::f64::consts;

/// For a given window size, generates the curve of the left-half of the window.
fn generate_win_curve(bs: usize) -> Vec<f32> {
    let len = bs / 2;
    let denom = f64::from(len as u32);

    let mut slope = vec![0.0; len];

    for (i, s) in slope.iter_mut().enumerate() {
        let num = f64::from(i as u32) + 0.5;
        let frac = consts::FRAC_PI_2 * (num / denom);
        *s = (consts::FRAC_PI_2 * frac.sin().powi(2)).sin() as f32
    }

    slope
}

pub struct Windows {
    /// Short block window left-half curve.
    pub short: Vec<f32>,
    /// `short` reversed: the right-half curve.
    pub short_rev: Vec<f32>,
    /// Long block window left-half curve.
    pub long: Vec<f32>,
    /// `long` reversed: the right-half curve.
    pub long_rev: Vec<f32>,
}

impl Windows {
    pub fn new(blocksize0: usize, blocksize1: usize) -> Self {
        let short = generate_win_curve(blocksize0);
        let long = generate_win_curve(blocksize1);
        // The overlap-add kernels read the falling slope forwards, so it is
        // stored rather than reversed per sample.
        let short_rev = short.iter().rev().copied().collect();
        let long_rev = long.iter().rev().copied().collect();
        Windows {
            short,
            short_rev,
            long,
            long_rev,
        }
    }
}
