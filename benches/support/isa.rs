//! Runs one workload once per instruction set, with a correctness guard.
//!
//! The rest of [`super`] answers "what is the input and how much work is it";
//! this module answers "which SIMD backend ran, and did it compute the same
//! thing". A group built through [`bench_across_isas`] does three things in
//! order:
//!
//! 1. runs the workload under every instruction set
//!    [`zvidlib::simd::available`] reports and checks that all of them produce
//!    byte-identical output (see [`assert_bit_exact_across_isas`]);
//! 2. asserts, through [`zvidlib::simd::active_by_site`], that each dispatch
//!    family really did follow the override;
//! 3. hands each arm to criterion as `<codec>/<isa>`, which is what makes
//!    scalar-vs-SIMD a direct criterion comparison rather than two unrelated
//!    runs.
//!
//! # Group naming
//!
//! These groups are named `<codec>/<isa>` (`av1_deblock/scalar` next to
//! `av1_deblock/neon`) rather than carrying the `simd=on`/`simd=off` build tag
//! [`super::group_name`] adds. The tag distinguishes two *builds* of a crate
//! whose kernels are chosen at run time; here the instruction set is the
//! measured axis and is named directly, so both arms of a comparison always
//! appear in the same run.

// Compiled separately by each bench target; `audio_decode.rs` has no
// scalar-vs-SIMD axis and uses none of this module.
#![allow(dead_code)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::Criterion;
use zvidlib::simd::{self, SimdIsa};

use super::FrameWork;

/// Logs what this host can actually execute, before anything is timed.
///
/// Every bench target starts with this, so a saved baseline names the arms it
/// was measured on. Groups built through [`bench_across_isas`] run one arm per
/// entry in [`simd::available`], so a runner without AVX2 simply has no `avx2`
/// arm rather than reporting scalar numbers under a vector label. That is the
/// right behaviour, but it is invisible in a results table: `av1_deblock/avx2`
/// being absent and `av1_deblock/avx2` being slow look the same from the
/// outside, and GitHub's runner pool is not uniform in AVX2 availability. CI
/// lifts these lines into its job summary so a run whose vector arms vanished
/// because the runner pool changed is diagnosable rather than mysterious.
///
/// [`simd::active_by_site`] is logged alongside it because "this host supports
/// AVX2" and "every dispatch family agrees to use it" are separate claims;
/// [`bench_across_isas`] asserts the second one per arm, and this prints its
/// starting state.
///
/// It takes a `&mut Criterion` and measures nothing so that it can be listed in
/// a target's `criterion_group!` like any other group, which is what guarantees
/// it runs before the first timed arm.
pub fn log_host_isas(_criterion: &mut Criterion) {
    let names: Vec<&str> = simd::available().iter().map(|isa| isa.name()).collect();
    println!("# host instruction sets: {}", names.join(", "));
    println!("# widest detected instruction set: {}", simd::active().name());
    for (site, isa) in simd::active_by_site() {
        println!("# dispatch site {site}: {}", isa.name());
    }
}

/// How to measure one workload across instruction sets.
pub struct IsaWorkload<'a> {
    /// Group name, conventionally `<codec>_<stage>` (e.g. `av1_deblock`).
    pub codec: &'a str,
    /// Frames and resolution one iteration covers, for throughput reporting.
    pub work: FrameWork,
    /// Criterion sample count. Codec workloads are far slower than criterion's
    /// default target, so callers lower this deliberately.
    pub sample_size: usize,
    /// Criterion measurement window per arm.
    pub measurement_time: Duration,
    /// Criterion warm-up window per arm.
    pub warm_up_time: Duration,
}

impl<'a> IsaWorkload<'a> {
    /// A workload whose defaults suit a slow, frame-scale codec routine.
    #[must_use]
    pub fn new(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
        IsaWorkload {
            codec,
            work,
            sample_size: 10,
            measurement_time: Duration::from_secs(5),
            warm_up_time: Duration::from_millis(500),
        }
    }
}

/// Benchmarks `run` once per available instruction set.
///
/// `run` must be deterministic and must return the bytes that identify its
/// result — decoded samples, filtered pixels, an encoded bitstream. Those bytes
/// are what the bit-exactness guard compares across arms, so returning
/// something that does not depend on the kernels under test would silently
/// disarm it.
pub fn bench_across_isas<F>(criterion: &mut Criterion, workload: &IsaWorkload<'_>, run: F)
where
    F: Fn() -> Vec<u8>,
{
    assert_bit_exact_across_isas(workload.codec, &run);

    let mut group = criterion.benchmark_group(workload.codec);
    group.sample_size(workload.sample_size);
    group.warm_up_time(workload.warm_up_time);
    group.measurement_time(workload.measurement_time);
    group.throughput(workload.work.elements());
    for isa in simd::available() {
        simd::set_override(Some(isa));
        assert_reached_every_site(workload.codec, isa);
        report_megapixels_per_second(workload, isa, &run);
        group.bench_function(isa.name(), |bencher| bencher.iter(|| black_box(run())));
    }
    simd::set_override(None);
    group.finish();
}

/// Asserts the override actually landed in every dispatch family.
///
/// This is the check that makes a scalar arm trustworthy. A timing difference
/// cannot distinguish "the override never reached this kernel" from "this
/// kernel's vector path is not faster on this host" — and the latter really
/// happens, notably for HEVC on hosts where the scalar reference
/// auto-vectorizes well under `lto = "fat"`. Reading each site's own selector
/// back through [`simd::active_by_site`] settles it directly.
pub fn assert_reached_every_site(codec: &str, isa: SimdIsa) {
    for (site, site_isa) in simd::active_by_site() {
        assert_eq!(
            site_isa,
            isa,
            "{codec}: pinning {} left the {site} kernels on {}",
            isa.name(),
            site_isa.name()
        );
    }
}

/// Times one pass of `run` and prints its megapixel throughput.
///
/// Criterion's own `Throughput::Elements(frames)` reports frames per second,
/// which is how a codec is usually specified but is not comparable across
/// resolutions. Megapixels per second is, and it is the unit the crate's
/// existing SIMD measurements are quoted in, so both are reported.
fn report_megapixels_per_second<F>(workload: &IsaWorkload<'_>, isa: SimdIsa, run: &F)
where
    F: Fn() -> Vec<u8>,
{
    let started = Instant::now();
    let output = black_box(run());
    let elapsed = started.elapsed();
    println!(
        "# {}/{}: {} frame(s)/iter, {:.4} Mpx in {:.4}s => {:.1} Mpx/s ({} output bytes)",
        workload.codec,
        isa.name(),
        workload.work.frames,
        workload.work.megapixels(),
        elapsed.as_secs_f64(),
        workload.work.megapixels_per_second(elapsed),
        output.len(),
    );
}

/// Checks that every available instruction set produces identical output.
///
/// Every SIMD backend in zvidlib is documented and tested as bit-exact with its
/// scalar reference. This re-checks that claim against the exact workload about
/// to be timed, because a speedup measured on a kernel that quietly diverged is
/// worse than no measurement: it looks like progress.
pub fn assert_bit_exact_across_isas<F>(label: &str, run: &F)
where
    F: Fn() -> Vec<u8>,
{
    simd::set_override(Some(SimdIsa::Scalar));
    let reference = run();
    let reference_sum = checksum(&reference);
    for isa in simd::available() {
        if isa == SimdIsa::Scalar {
            continue;
        }
        simd::set_override(Some(isa));
        assert_reached_every_site(label, isa);
        let actual = run();
        assert_eq!(
            actual.len(),
            reference.len(),
            "{label}: {} produced {} output bytes, scalar produced {}",
            isa.name(),
            actual.len(),
            reference.len()
        );
        assert_eq!(
            checksum(&actual),
            reference_sum,
            "{label}: {} output diverged from the scalar reference",
            isa.name()
        );
    }
    simd::set_override(None);
}

/// FNV-1a over a byte buffer.
///
/// Used only to compare one backend's output against another's, so it needs to
/// be cheap and order-sensitive, not cryptographic.
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
