//! Runs one workload once per instruction set, with a correctness guard.
//!
//! A SIMD benchmark is only meaningful if the arms it compares compute the
//! same thing. [`bench_across_isas`] therefore does three things in order:
//!
//! 1. runs the workload under every instruction set
//!    [`zvidlib::simd::available`] reports and checks that all of them produce
//!    byte-identical output (see [`assert_bit_exact_across_isas`]);
//! 2. reports megapixels per second for each arm from a single timed pass, so
//!    the numbers are comparable to the throughput figures in the changelog;
//! 3. hands each arm to criterion as `<codec>/<isa>` inside one benchmark
//!    group, which is what makes scalar-vs-SIMD a direct criterion comparison
//!    rather than two unrelated runs.
//!
//! The override is always cleared afterwards, including when a workload
//! panics is *not* guaranteed — a panicking bench aborts the run anyway, so
//! there is no state left to observe.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput};
use zvidlib::simd::{self, SimdIsa};

use super::checksum;

/// How to measure one workload.
pub struct Workload<'a> {
    /// Group name, conventionally `<codec>_<stage>` (e.g. `hevc_decode`).
    pub codec: &'a str,
    /// Frames one iteration of the workload processes. Reported to criterion
    /// as [`Throughput::Elements`], so criterion prints frames per second.
    pub frames_per_iteration: u64,
    /// Luma samples per frame, used for the megapixels-per-second line.
    pub pixels_per_frame: u64,
    /// Criterion sample count. The codec workloads here are far slower than
    /// criterion's default target, so they lower this deliberately.
    pub sample_size: usize,
    /// Criterion measurement window per arm.
    pub measurement_time: Duration,
    /// Criterion warm-up window per arm.
    pub warm_up_time: Duration,
}

impl Workload<'_> {
    /// A workload whose defaults suit a slow, frame-scale codec routine.
    #[must_use]
    pub fn new(codec: &str, frames_per_iteration: u64, pixels_per_frame: u64) -> Workload<'_> {
        Workload {
            codec,
            frames_per_iteration,
            pixels_per_frame,
            sample_size: 10,
            measurement_time: Duration::from_secs(5),
            warm_up_time: Duration::from_millis(500),
        }
    }
}

/// Benchmarks `run` once per available instruction set.
///
/// `run` must be deterministic and must return the bytes that identify its
/// result — decoded samples, filtered pixels, an encoded bitstream. Those
/// bytes are what the bit-exactness guard compares across arms, so returning
/// something that does not depend on the kernels under test would silently
/// disarm it.
pub fn bench_across_isas<F>(criterion: &mut Criterion, workload: &Workload<'_>, run: F)
where
    F: Fn() -> Vec<u8>,
{
    assert_bit_exact_across_isas(workload.codec, &run);

    let mut group = criterion.benchmark_group(workload.codec);
    group.sample_size(workload.sample_size);
    group.warm_up_time(workload.warm_up_time);
    group.measurement_time(workload.measurement_time);
    group.throughput(Throughput::Elements(workload.frames_per_iteration));
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
fn assert_reached_every_site(codec: &str, isa: SimdIsa) {
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
/// which is the number a codec is usually specified in, but it is not
/// comparable across resolutions. Megapixels per second is, and it is the unit
/// the crate's existing SIMD measurements are quoted in, so both are reported.
fn report_megapixels_per_second<F>(workload: &Workload<'_>, isa: SimdIsa, run: &F)
where
    F: Fn() -> Vec<u8>,
{
    let started = Instant::now();
    let output = black_box(run());
    let elapsed = started.elapsed().as_secs_f64();
    let megapixels =
        (workload.frames_per_iteration * workload.pixels_per_frame) as f64 / 1_000_000.0;
    let rate = if elapsed > 0.0 {
        megapixels / elapsed
    } else {
        f64::INFINITY
    };
    eprintln!(
        "{}/{}: {} frame(s), {megapixels:.2} Mpx in {elapsed:.4}s => {rate:.1} Mpx/s ({} output bytes)",
        workload.codec,
        isa.name(),
        workload.frames_per_iteration,
        output.len(),
    );
}

/// Checks that every available instruction set produces identical output.
///
/// Every SIMD backend in zvidlib is documented and tested as bit-exact with
/// its scalar reference. This re-checks that claim against the exact workload
/// about to be timed, because a speedup measured on a kernel that quietly
/// diverged is worse than no measurement: it looks like progress.
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
