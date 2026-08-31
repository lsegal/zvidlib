//! Scalar-versus-SIMD benchmarks for zvidlib's AV1 encoder-side kernels.
//!
//! The AV1 encoder's forward transforms are the counterpart to the inverse
//! transforms `benches/av1_decode.rs` measures: the same block sizes, the same
//! `Av1TxType` families, and the same `av1_simd` dispatch site, run in the
//! encoding direction. They are measured here rather than beside the inverse
//! sweep because they are encoder work, and a decoder target that reports
//! encoder numbers is a target whose scope cannot be read off its name.
//!
//! Every group runs once per instruction set `zvidlib::simd::available()`
//! reports, through the crate-wide override in [`zvidlib::simd`], and
//! `benches/support/isa.rs` asserts that each arm is bit-exact with scalar
//! before timing it and that the override really landed in each dispatch
//! family — so a reported speedup cannot come from a kernel that quietly
//! diverged or from a switch that never took effect.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `av1_forward_dct_{4x4,8x8,16x16,32x32}` | forward DCT, `src/av1_encoder/transform.rs` through `zvidlib::forward_transform` |
//! | `av1_forward_adst_8x8`, `av1_forward_flipadst_16x16` | the forward ADST family, including a flipped type |
//!
//! The block counts and the coefficient generator are the ones the groups were
//! introduced with (issue #140, in `tests/av1_simd_bench.rs`, and then in
//! `benches/av1_decode.rs`), so the numbers stay directly comparable with the
//! inverse-transform groups and with everything reported for them before the
//! move.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::{Av1TxType, forward_transform};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, checksum, log_host_isas};

/// Luma dimensions the kernel-level groups run over, matching
/// `benches/av1_decode.rs`. One 1080p plane is large enough that per-call
/// dispatch overhead is negligible next to the vectorized inner loops, and it
/// is the size these measurements have always used, so their numbers stay
/// comparable across the move.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the kernel groups.
///
/// Each group is measured once per available instruction set, so the default
/// five-second window would stretch a plain `cargo bench --bench av1_encode`
/// out for no extra resolution. Two seconds over a 1080p plane is still
/// hundreds of iterations of work per sample.
fn kernel_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

// ---------------------------------------------------------------------------
// Forward transforms (src/av1_encoder/transform.rs, src/av1_simd/transforms.rs)
// ---------------------------------------------------------------------------

/// Every forward transform size and family the vector kernels cover, applied
/// over a whole frame's worth of blocks.
fn av1_forward_transforms(criterion: &mut Criterion) {
    for (name, size, tx_type) in [
        ("av1_forward_dct_4x4", 4usize, Av1TxType::DctDct),
        ("av1_forward_dct_8x8", 8, Av1TxType::DctDct),
        ("av1_forward_dct_16x16", 16, Av1TxType::DctDct),
        ("av1_forward_dct_32x32", 32, Av1TxType::DctDct),
        ("av1_forward_adst_8x8", 8, Av1TxType::AdstAdst),
        ("av1_forward_flipadst_16x16", 16, Av1TxType::FlipadstAdst),
    ] {
        let residual: Vec<i32> = (0..size * size)
            .map(|index| (index as i32 * 53) % 511 - 255)
            .collect();
        let blocks = (WIDTH / size) * (HEIGHT / size);
        let covered_width = (WIDTH / size * size) as u64;
        let covered_height = (HEIGHT / size * size) as u64;
        let work = FrameWork::new(1, covered_width, covered_height);
        let workload = kernel_workload(name, work);
        bench_across_isas(criterion, &workload, || {
            let mut digest = 0u64;
            for _ in 0..blocks {
                let coefficients = forward_transform(&residual, size, tx_type);
                digest ^= checksum(&coefficients[0].to_le_bytes());
            }
            digest.to_le_bytes().to_vec()
        });
    }
}

criterion_group!(benches, log_host_isas, av1_forward_transforms);
criterion_main!(benches);
