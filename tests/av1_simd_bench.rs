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
    let mut deblocking = Vec::new();
    let mut cdef = Vec::new();
    let mut wiener = Vec::new();
    let mut self_guided = Vec::new();
    let mut dct4 = Vec::new();
    let mut dct8 = Vec::new();
    let mut dct16 = Vec::new();
    let mut dct32 = Vec::new();
    let mut adst8 = Vec::new();
    let mut adst16 = Vec::new();

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

        // One inverse transform per block of the frame, at every size and
        // transform type the vector kernels cover.
        for (slot, size, tx_type) in [
            (&mut dct4, 4usize, Av1TxType::DctDct),
            (&mut dct8, 8, Av1TxType::DctDct),
            (&mut dct16, 16, Av1TxType::DctDct),
            (&mut dct32, 32, Av1TxType::DctDct),
            (&mut adst8, 8, Av1TxType::AdstAdst),
            (&mut adst16, 16, Av1TxType::FlipadstAdst),
        ] {
            let coefficients: Vec<i32> = (0..size * size)
                .map(|index| (index as i32 * 37) % 121 - 60)
                .collect();
            let blocks = (WIDTH / size) * (HEIGHT / size);
            slot.push((
                isa,
                measure(|| {
                    let mut checksum = 0i64;
                    for _ in 0..blocks {
                        let residual = inverse_transform(&coefficients, size, tx_type, 20, 14);
                        checksum += i64::from(residual[0]);
                    }
                    assert_ne!(checksum, i64::MIN);
                }),
            ));
        }
    }
    set_active_isa(None);

    println!(
        "{:<28} {:<8} {:>12}  {:>6}",
        "stage", "isa", "time", "vs scalar"
    );
    report("deblocking (frame)", &deblocking);
    report("cdef (frame)", &cdef);
    report("wiener restoration (frame)", &wiener);
    report("self-guided (256x256 unit)", &self_guided);
    report("inverse dct 4x4 (frame)", &dct4);
    report("inverse dct 8x8 (frame)", &dct8);
    report("inverse dct 16x16 (frame)", &dct16);
    report("inverse dct 32x32 (frame)", &dct32);
    report("inverse adst 8x8 (frame)", &adst8);
    report("flip-adst 16x16 (frame)", &adst16);
}
