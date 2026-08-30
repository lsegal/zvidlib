//! Bit-exactness and throughput coverage for the SIMD AV1 intra prediction
//! kernels (`zvidlib::av1_intra_pred`).
//!
//! Every test compares the dispatched kernels (AVX2/SSE4.1 on x86_64, NEON on
//! aarch64, scalar elsewhere) against an independent scalar reference written
//! here, so a vector path that disagrees with the spec-derived scalar
//! definition fails regardless of which host runs the suite.

use std::time::Instant;

use zvidlib::{
    Av1IntraBlock, Av1IntraFrame, Av1IntraMode, ColorRange, Limits, VideoDimensions,
    add_residual_row, av1_intra_simd, paeth_row, sum_samples,
};

const MODES: [Av1IntraMode; 4] = [
    Av1IntraMode::Dc,
    Av1IntraMode::Vertical,
    Av1IntraMode::Horizontal,
    Av1IntraMode::Paeth,
];

/// AV1 block dimensions the reconstruction path sees, from the smallest
/// transform block up to the largest square superblock partition.
const SIZES: [usize; 6] = [4, 8, 16, 32, 64, 128];

fn pseudo_random(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

fn reference_paeth(top_left: u8, top: u8, left: u8) -> u8 {
    let base = i16::from(top) + i16::from(left) - i16::from(top_left);
    let distance = |candidate: u8| (base - i16::from(candidate)).unsigned_abs();
    if distance(top) <= distance(left) && distance(top) <= distance(top_left) {
        top
    } else if distance(left) <= distance(top_left) {
        left
    } else {
        top_left
    }
}

fn reference_add_residual(residuals: &[i16], row: &mut [u8]) {
    for (sample, &residual) in row.iter_mut().zip(residuals) {
        *sample = i16::from(*sample).saturating_add(residual).clamp(0, 255) as u8;
    }
}

/// The pre-SIMD `Av1IntraFrame::reconstruct_block` inner loop, kept here as
/// the reference the vectorized reconstruction must match sample for sample.
fn reference_reconstruct(plane: &mut [u8], stride: usize, block: Av1IntraBlock, residuals: &[i16]) {
    let (x, y, width, height) = (block.x, block.y, block.width, block.height);
    let top_left = if x > 0 && y > 0 {
        plane[(y - 1) * stride + x - 1]
    } else {
        128
    };
    let mut top = vec![128u8; width];
    let mut left = vec![128u8; height];
    if y > 0 {
        for (i, sample) in top.iter_mut().enumerate() {
            *sample = plane[(y - 1) * stride + x + i];
        }
    }
    if x > 0 {
        for (i, sample) in left.iter_mut().enumerate() {
            *sample = plane[(y + i) * stride + x - 1];
        }
    }
    let dc = (top.iter().map(|&v| u32::from(v)).sum::<u32>()
        + left.iter().map(|&v| u32::from(v)).sum::<u32>())
        / u32::try_from(width + height).expect("nonzero block dimensions");
    for row in 0..height {
        for column in 0..width {
            let prediction = match block.mode {
                Av1IntraMode::Dc => dc as u8,
                Av1IntraMode::Vertical => top[column],
                Av1IntraMode::Horizontal => left[row],
                Av1IntraMode::Paeth => reference_paeth(top_left, top[column], left[row]),
            };
            plane[(y + row) * stride + x + column] = i16::from(prediction)
                .saturating_add(residuals[row * width + column])
                .clamp(0, 255) as u8;
        }
    }
}

#[test]
fn sum_samples_matches_the_scalar_reference_for_every_length() {
    let mut seed = 3;
    let samples: Vec<u8> = (0..300)
        .map(|_| (pseudo_random(&mut seed) >> 12) as u8)
        .collect();
    for length in 0..samples.len() {
        let expected: u32 = samples[..length].iter().map(|&s| u32::from(s)).sum();
        assert_eq!(sum_samples(&samples[..length]), expected, "length {length}");
    }
}

#[test]
fn residual_and_paeth_kernels_match_the_scalar_reference() {
    let mut seed = 17;
    for length in 0..96usize {
        let row: Vec<u8> = (0..length)
            .map(|_| (pseudo_random(&mut seed) >> 10) as u8)
            .collect();
        let residuals: Vec<i16> = (0..length)
            .map(|_| (pseudo_random(&mut seed) % 8192) as i16 - 4096)
            .collect();
        let mut expected = row.clone();
        reference_add_residual(&residuals, &mut expected);
        let mut actual = row.clone();
        add_residual_row(&residuals, &mut actual);
        assert_eq!(actual, expected, "add_residual_row length {length}");

        let top_left = (pseudo_random(&mut seed) >> 7) as u8;
        let left = (pseudo_random(&mut seed) >> 7) as u8;
        let expected: Vec<u8> = row
            .iter()
            .map(|&above| reference_paeth(top_left, above, left))
            .collect();
        let mut actual = vec![0u8; length];
        paeth_row(top_left, &row, left, &mut actual);
        assert_eq!(actual, expected, "paeth_row length {length}");
    }
}

#[test]
fn reconstructed_blocks_are_bit_exact_for_every_size_and_mode() {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(256, 256, &limits).expect("valid dimensions");
    let mut seed = 91;
    for mode in MODES {
        for size in SIZES {
            for &(x, y) in &[(0usize, 0usize), (0, size), (size, 0), (size, size)] {
                let luma: Vec<u8> = (0..256 * 256)
                    .map(|_| (pseudo_random(&mut seed) >> 9) as u8)
                    .collect();
                let residuals: Vec<i16> = (0..size * size)
                    .map(|_| (pseudo_random(&mut seed) % 2048) as i16 - 1024)
                    .collect();
                let block = Av1IntraBlock {
                    plane: 0,
                    x,
                    y,
                    width: size,
                    height: size,
                    mode,
                };

                let mut expected = luma.clone();
                reference_reconstruct(&mut expected, 256, block, &residuals);

                let mut frame =
                    Av1IntraFrame::from_luma(dimensions, luma, ColorRange::Limited, &limits)
                        .expect("luma frame");
                frame
                    .reconstruct_block(block, &residuals)
                    .expect("reconstructed block");
                let video = frame.into_video_frame(&limits).expect("video frame");
                assert_eq!(
                    video.planes[0].data, expected,
                    "mode {mode:?} size {size} at ({x}, {y})"
                );
            }
        }
    }
}

/// Reports the throughput of the dispatched kernels against the scalar
/// reference for representative AV1 block sizes. Run with
/// `cargo test --release --features native --test av1_simd_intra -- --nocapture`
/// to see the numbers; the test itself only asserts that both paths agree, so
/// it never fails because a shared CI runner was slow.
#[test]
fn benchmark_intra_kernels_against_the_scalar_reference() {
    /// Passes over the working set per measurement, after one warm-up pass.
    const PASSES: u32 = 32;

    fn measure(passes: u32, mut body: impl FnMut()) -> f64 {
        body();
        let started = Instant::now();
        for _ in 0..passes {
            body();
        }
        started.elapsed().as_secs_f64() / f64::from(passes)
    }

    let mut seed = 5;
    println!("AV1 intra prediction kernels: {:?}", av1_intra_simd());
    for size in SIZES {
        let rows = 4096;
        let source: Vec<u8> = (0..size * rows)
            .map(|_| (pseudo_random(&mut seed) >> 13) as u8)
            .collect();
        let residuals: Vec<i16> = (0..size * rows)
            .map(|_| (pseudo_random(&mut seed) % 1024) as i16 - 512)
            .collect();
        let top_left = 96u8;

        let mut scalar = source.clone();
        let scalar_add = measure(PASSES, || {
            scalar.copy_from_slice(&source);
            for row in 0..rows {
                reference_add_residual(
                    &residuals[row * size..(row + 1) * size],
                    &mut scalar[row * size..(row + 1) * size],
                );
            }
        });
        let mut simd = source.clone();
        let simd_add = measure(PASSES, || {
            simd.copy_from_slice(&source);
            for row in 0..rows {
                add_residual_row(
                    &residuals[row * size..(row + 1) * size],
                    &mut simd[row * size..(row + 1) * size],
                );
            }
        });
        assert_eq!(simd, scalar, "residual add mismatch at size {size}");

        let mut scalar_predictions = vec![0u8; size * rows];
        let scalar_paeth = measure(PASSES, || {
            for row in 0..rows {
                let left = source[row * size];
                for column in 0..size {
                    scalar_predictions[row * size + column] =
                        reference_paeth(top_left, source[row * size + column], left);
                }
            }
        });
        let mut simd_predictions = vec![0u8; size * rows];
        let simd_paeth = measure(PASSES, || {
            for row in 0..rows {
                let left = source[row * size];
                paeth_row(
                    top_left,
                    &source[row * size..(row + 1) * size],
                    left,
                    &mut simd_predictions[row * size..(row + 1) * size],
                );
            }
        });
        assert_eq!(
            simd_predictions, scalar_predictions,
            "paeth mismatch at size {size}"
        );

        let samples = (size * rows) as f64;
        println!(
            "{size:>3}x{size:<3} residual add {:>6.2} -> {:>6.2} ns/sample ({:.2}x)   paeth {:>6.2} -> {:>6.2} ns/sample ({:.2}x)",
            scalar_add * 1e9 / samples,
            simd_add * 1e9 / samples,
            scalar_add / simd_add,
            scalar_paeth * 1e9 / samples,
            simd_paeth * 1e9 / samples,
            scalar_paeth / simd_paeth,
        );
    }
}

/// End-to-end `Av1IntraFrame::reconstruct_block` throughput against the
/// pre-SIMD reference loop, for the block sizes AV1 reconstruction uses.
#[test]
fn benchmark_block_reconstruction_against_the_scalar_reference() {
    const PASSES: u32 = 16;
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(512, 512, &limits).expect("valid dimensions");
    let mut seed = 41;
    let luma: Vec<u8> = (0..512 * 512)
        .map(|_| (pseudo_random(&mut seed) >> 13) as u8)
        .collect();

    for mode in MODES {
        for size in [8usize, 16, 32, 64] {
            let residuals: Vec<i16> = (0..size * size)
                .map(|_| (pseudo_random(&mut seed) % 1024) as i16 - 512)
                .collect();
            let blocks: Vec<Av1IntraBlock> = (0..512 / size)
                .flat_map(|row| {
                    (0..512 / size).map(move |column| Av1IntraBlock {
                        plane: 0,
                        x: column * size,
                        y: row * size,
                        width: size,
                        height: size,
                        mode,
                    })
                })
                .collect();

            let mut reference = luma.clone();
            let started = Instant::now();
            for _ in 0..PASSES {
                for &block in &blocks {
                    reference_reconstruct(&mut reference, 512, block, &residuals);
                }
            }
            let scalar = started.elapsed().as_secs_f64() / f64::from(PASSES);

            let mut frame =
                Av1IntraFrame::from_luma(dimensions, luma.clone(), ColorRange::Limited, &limits)
                    .expect("luma frame");
            let started = Instant::now();
            for _ in 0..PASSES {
                for &block in &blocks {
                    frame.reconstruct_block(block, &residuals).expect("block");
                }
            }
            let simd = started.elapsed().as_secs_f64() / f64::from(PASSES);

            println!(
                "{mode:?} {size}x{size}: scalar {:>8.3?} ms, simd {:>8.3?} ms ({:.2}x)",
                scalar * 1e3,
                simd * 1e3,
                scalar / simd,
            );
        }
    }
}
