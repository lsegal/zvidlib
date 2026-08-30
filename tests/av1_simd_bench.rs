//! Throughput benchmark for the AV1 SIMD kernels (issue #120).
//!
//! Ignored by default because it is a timing measurement, not a correctness
//! check (`tests` in `src/av1_simd/tests.rs` covers bit-exactness). Run it with:
//!
//! ```text
//! cargo test --release --features native --test av1_simd_bench -- --ignored --nocapture
//! ```
//!
//! Every stage is measured over a representative 1920x1080 luma frame, once per
//! instruction set the host supports, and reported relative to the scalar path.

use std::time::{Duration, Instant};

use zvidlib::Limits;
use zvidlib::TxSizeGrid;
use zvidlib::av1_filters::{
    CdefStrength, FilterFrame, FilterPlane, LoopFilterParams, RestorationUnit,
    apply_restoration_unit, cdef_frame, deblock_frame,
};
use zvidlib::av1_intra::{Av1TxType, inverse_transform};
use zvidlib::av1_simd::{SimdIsa, available_isas, set_active_isa};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Deterministic pseudo-random frame content with enough local structure that
/// the filters' data-dependent branches are actually taken.
fn representative_plane() -> FilterPlane {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
    for index in 0..WIDTH * HEIGHT {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (state >> 56) as i32 & 0x1f;
        let gradient = ((index % WIDTH) / 8 + (index / WIDTH) / 8) as i32;
        data.push(((gradient + noise) & 0xff) as u8);
    }
    FilterPlane::from_samples(WIDTH, HEIGHT, data, &Limits::default()).unwrap()
}

/// Near-flat block content: the wide (8-tap and 14-tap) deblocking filters are
/// gated on a flatness check, so they only do work on content like this.
fn flat_blocks_plane(width: usize, height: usize) -> FilterPlane {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut data = Vec::with_capacity(width * height);
    for index in 0..width * height {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let (x, y) = (index % width, index / width);
        let block = ((x / 32 + y / 32) % 5) as i32;
        data.push((100 + block * 6 + ((state >> 60) as i32 & 1)) as u8);
    }
    FilterPlane::from_samples(width, height, data, &Limits::default()).unwrap()
}

/// A frame-wide grid of 32x32 transform blocks, which is what makes every luma
/// edge select the 14-tap filter (spec §7.14.5).
fn wide_tx_grid(width: usize, height: usize) -> TxSizeGrid {
    let mut grid = TxSizeGrid::new(width, height);
    for y in (0..height).step_by(32) {
        for x in (0..width).step_by(32) {
            grid.set_block(x, y, 32, 32);
        }
    }
    grid
}

fn measure(mut body: impl FnMut()) -> Duration {
    body();
    let start = Instant::now();
    body();
    start.elapsed()
}

fn report(stage: &str, timings: &[(SimdIsa, Duration)]) {
    let (_, baseline) = timings[0];
    for (isa, elapsed) in timings {
        let speedup = baseline.as_secs_f64() / elapsed.as_secs_f64();
        println!(
            "{stage:<28} {:<8} {:>9.2} ms  {:>5.2}x",
            isa.name(),
            elapsed.as_secs_f64() * 1000.0,
            speedup
        );
    }
}

#[test]
#[ignore = "timing benchmark; run explicitly with --ignored --nocapture"]
fn av1_simd_speedup_on_a_representative_frame() {
    let isas = available_isas();
    println!("host instruction sets: {:?}", isas);
    println!("frame: {WIDTH}x{HEIGHT} luma\n");

    let source = representative_plane();
    let flat = flat_blocks_plane(WIDTH, HEIGHT);
    let wide_grid = wide_tx_grid(WIDTH, HEIGHT);
    // Small planes are where boundary positions dominate: a 33x17 plane is
    // almost entirely frame border and partial rows/columns.
    let small: Vec<FilterPlane> = (0..64).map(|_| flat_blocks_plane(33, 17)).collect();
    let mut deblocking = Vec::new();
    let mut deblocking_wide = Vec::new();
    let mut deblocking_small = Vec::new();
    let mut cdef = Vec::new();
    let mut wiener = Vec::new();
    let mut self_guided = Vec::new();
    let mut transforms = Vec::new();

    for isa in isas {
        set_active_isa(Some(isa));

        let params = LoopFilterParams {
            y_vertical_level: 32,
            y_horizontal_level: 32,
            u_level: 0,
            v_level: 0,
            sharpness: 0,
        };
        let mut frame = FilterFrame::new_monochrome(source.clone());
        deblocking.push((
            isa,
            measure(|| {
                deblock_frame(&mut frame, &params, None).unwrap();
            }),
        ));

        let strength = CdefStrength {
            y_primary: 4,
            y_secondary: 2,
            uv_primary: 0,
            uv_secondary: 0,
            damping: 3,
        };
        let mut frame = FilterFrame::new_monochrome(source.clone());
        cdef.push((
            isa,
            measure(|| {
                cdef_frame(&mut frame, &strength, &Limits::default()).unwrap();
            }),
        ));

        let mut plane = source.clone();
        let unit = RestorationUnit::Wiener {
            horizontal: [3, -7, 15],
            vertical: [-2, 5, 11],
        };
        wiener.push((
            isa,
            measure(|| {
                apply_restoration_unit(&mut plane, &unit, 0, 0, WIDTH, HEIGHT).unwrap();
            }),
        ));

        // Self-guided restoration is signaled per restoration unit rather than
        // per frame, so this measures one 256x256 unit's worth of work.
        let mut plane = source.clone();
        let unit = RestorationUnit::SelfGuided {
            radius: [2, 3],
            eps: [12, 30],
            weight: [40, 24],
        };
        self_guided.push((
            isa,
            measure(|| {
                apply_restoration_unit(&mut plane, &unit, 0, 0, 256, 256).unwrap();
            }),
        ));

        let mut frame = FilterFrame::new_monochrome(flat.clone());
        deblocking_wide.push((
            isa,
            measure(|| {
                deblock_frame(&mut frame, &params, Some(&wide_grid)).unwrap();
            }),
        ));

        let mut planes = small.clone();
        deblocking_small.push((
            isa,
            measure(|| {
                for plane in planes.iter_mut() {
                    let mut frame = FilterFrame::new_monochrome(plane.clone());
                    deblock_frame(&mut frame, &params, None).unwrap();
                    *plane = frame.y;
                }
            }),
        ));

        // One inverse transform per 8x8 block of the frame.
        let coefficients: Vec<i32> = (0..64).map(|index: i32| (index * 37) % 121 - 60).collect();
        let blocks = (WIDTH / 8) * (HEIGHT / 8);
        transforms.push((
            isa,
            measure(|| {
                let mut checksum = 0i64;
                for _ in 0..blocks {
                    let residual = inverse_transform(&coefficients, 8, Av1TxType::DctDct, 20, 14);
                    checksum += i64::from(residual[0]);
                }
                assert_ne!(checksum, i64::MIN);
            }),
        ));
    }
    set_active_isa(None);

    println!(
        "{:<28} {:<8} {:>12}  {:>6}",
        "stage", "isa", "time", "vs scalar"
    );
    report("deblocking (frame)", &deblocking);
    report("deblocking (frame, 14-tap)", &deblocking_wide);
    report("deblocking (64x 33x17 planes)", &deblocking_small);
    report("cdef (frame)", &cdef);
    report("wiener restoration (frame)", &wiener);
    report("self-guided (256x256 unit)", &self_guided);
    report("inverse dct 8x8 (frame)", &transforms);
}
