# Benchmarks

zvidlib's benchmarks run under [criterion](https://docs.rs/criterion) with
`harness = false`. The whole suite lives in a single bench target
(`benches/codec.rs`) so the shared fixture cache in `benches/support/` is loaded
and decoded once per process and each iteration measures codec work only.

## Running

```sh
cargo bench                       # the default, fast groups
cargo bench --features simd       # the same groups, recorded under `simd=on`
cargo bench --no-run              # compile only
```

Criterion writes HTML reports to `target/criterion/report/index.html`.

### Filtering to one group or benchmark

Criterion treats its first positional argument as a regular expression matched
against the full benchmark id:

```sh
cargo bench --bench codec -- av1_decode          # one codec's group
cargo bench --bench codec -- 'simd=off'          # one feature arm
cargo bench --bench codec -- inter_show_existing # one benchmark
```

Use `--warm-up-time` and `--measurement-time` for a quicker smoke run:

```sh
cargo bench --bench codec -- --warm-up-time 0.5 --measurement-time 2
```

### The long-running 1080p group

The bundled `examples/media/BigBuckBunny.mp4` sample is 768 frames of 1920x1080
decoded by a pure-Rust decoder, so it is not part of the default run. Opt in with
an environment variable:

```sh
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench codec -- hevc_decode_1080p
```

## Group naming and the `simd` feature

`simd` is an additive, off-by-default cargo feature. The crate's vector kernels
are selected by runtime CPU feature detection, so the feature gates no code
today; it exists as the switch the per-codec benchmark work codes against.

Every criterion group name ends in the arm it was measured under —
`hevc_decode_1080p/simd=off` versus `hevc_decode_1080p/simd=on` — so the two
builds record separately and stay comparable in one report.

## Scalar versus SIMD

The `simd` feature above distinguishes two *builds*. The instruction set is a
separate, finer axis, and it is the one a SIMD comparison actually turns on:
zvidlib's kernels are chosen by runtime CPU feature detection, so scalar and
vector arms can both be measured in a single run.

`zvidlib::simd` is the process-wide switch that makes that possible:

```rust
use zvidlib::simd::{self, SimdIsa};

simd::set_override(Some(SimdIsa::Scalar)); // every kernel, HEVC and AV1
// ... run the workload ...
simd::set_override(None);                  // back to per-host detection
```

`set_override` reaches every dispatch family at once: the AV1 transforms and
in-loop filters, AV1 motion compensation, AV1 intra prediction, and every HEVC
engine kernel (inter/intra prediction, in-loop filters, inverse transforms, and
encoder-side distortion metrics). An instruction set this host cannot execute is
clamped to `SimdIsa::Scalar` rather than silently ignored, so the arm you asked
for is always a defined one. `simd::active()` reports what is in force and
`simd::available()` lists what this host can run.

Groups built through `support::isa::bench_across_isas` run once per entry in
`simd::available()` and are named `<codec>/<isa>`, so criterion compares the
arms directly:

```
av1_deblock/scalar
av1_deblock/neon
av1_motion_compensation/scalar
av1_motion_compensation/neon
```

On an `x86_64` host with AVX2 the arms are `scalar`, `sse4.1`, and `avx2`
instead. These groups deliberately carry the instruction set in the name rather
than the `simd=on`/`simd=off` build tag: the instruction set *is* the measured
axis here, and both arms always appear in the same run.

Filter to one of them the same way as any other group:

```sh
cargo bench --bench codec -- av1_deblock
cargo bench --bench codec -- 'av1_deblock/scalar'
```

The per-ISA HEVC group decodes the bundled 1080p sample, so it sits behind the
same `ZVIDLIB_BENCH_LARGE=1` opt-in as the other 1080p group.

### Correctness guard

Every per-ISA group runs `assert_bit_exact_across_isas` before timing anything:
each arm's output must be byte-identical to the scalar arm's. This is not
optional ceremony. Every vector backend in the crate is documented as bit-exact
with its scalar reference, and a speedup produced by a kernel that quietly
diverged would look like progress. If a backend disagrees, the benchmark panics
instead of reporting a number.

### Reading a null result

`simd::active_by_site()` reports what each dispatch family resolved to
individually:

```rust
for (site, isa) in zvidlib::simd::active_by_site() {
    println!("{site}: {}", isa.name());
}
```

The harness asserts on this before every timed arm, and you should too if you
are interpreting a surprising result. **A timing difference is not proof the
switch landed, and the absence of one is not proof it did not.** Some kernels
are near parity with their scalar reference on some hosts — the HEVC arms come
out roughly even on Apple Silicon, where LLVM auto-vectorizes the scalar code
well under `lto = "fat"`, while AV1 deblocking and motion compensation on the
same host are 2.4-4.9x. `active_by_site()` answers the question directly.

## Continuous integration

The `Benchmarks` job in `.github/workflows/ci.yml` treats the suite as two
different things depending on the event.

**On every pull request** it runs `cargo bench --no-run` and stops. That is one
build and no measurement. It exists because the usual way a benchmark suite dies
is not a bad number, it is rotting: the bench code stops compiling against the
crate, nobody runs it locally, and the decay is only discovered when someone
needs a measurement months later. Compiling on every PR makes that failure
immediate and cheap.

It deliberately does **not** time anything on a pull request. GitHub's shared
runners differ in CPU model, neighbour load, and thermal state between two runs
of the same commit by far more than the regressions worth catching. A PR gate on
those timings would fail on noise, and a check that fails on noise gets disabled.

**On `main` pushes and `workflow_dispatch`** it runs the full suite with
`ZVIDLIB_BENCH_LARGE=1`, so the per-ISA HEVC group — the one that proves the
crate-wide override reaches the HEVC kernels — is included. Then it:

1. writes `bench.log` and puts the host's instruction sets into the job summary;
2. reduces `target/criterion/` to one small JSON baseline through
   `.github/scripts/criterion_baseline.py collect`;
3. downloads the newest baseline artifact from a previous `main` run and diffs
   the two with `criterion_baseline.py compare`, writing a per-group delta table
   to the job summary;
4. uploads its own baseline as the `criterion-baseline-main` artifact (90-day
   retention) for the next run to compare against.

`workflow_dispatch` is how to measure a branch on demand without merging it, and
it takes a `threshold` input.

### The threshold, and why it is only a report

The comparison flags anything that moved more than **15%** and writes it into
the job summary. It does not fail the job.

15% is deliberately loose and deliberately provisional. Nobody knows this
suite's real run-to-run variance on the GitHub runner pool yet, and a threshold
guessed tighter than the noise floor produces false regressions immediately —
which is the same failure mode as gating PRs on timings, just slower. The
intended path is to leave it reporting for a few weeks, read the actual spread
off successive `main` runs, and then tighten it, and only then consider making it
fail. Raising the alarm before the alarm is calibrated trains everyone to ignore
it.

The comparison uses criterion's **median** point estimate rather than the mean,
because one descheduled iteration on a shared runner moves the mean and leaves
the median alone. It compares point estimates rather than running criterion's
own change detection, which needs both runs' raw sample data in one
`target/criterion/` directory and assumes the same machine produced both.

### Reading a flagged delta

A red row is a prompt to look, not a verdict. Successive `main` runs land on
different physical machines, and an arm can also disappear entirely — see the
instruction-set log below. Reproduce on a quiet host before treating a flagged
row as a real regression, and note that the crate's own measurements are quoted
from an Apple Silicon host while CI measures `x86_64` Linux, so the two are not
directly comparable to each other either.

### Why the host's instruction sets are logged

Groups built through `bench_across_isas` run one arm per entry in
`simd::available()`, so a runner without AVX2 simply has no `avx2` arm. That is
the correct behaviour — the alternative is scalar numbers filed under a vector
label — but it is invisible in a results table: an absent `av1_deblock/avx2` and
a slow one look the same from the outside, and GitHub's runner pool is not
uniform in AVX2 availability. The bench target therefore prints
`simd::available()`, the widest detected instruction set, and
`simd::active_by_site()` before anything is timed, and the job lifts those lines
into its summary. A baseline is only comparable to another baseline measured on
the same arms.

## Committed baselines

The numbers below are a reference point for the ratios between arms, not a
threshold anything is checked against — the CI job compares each `main` run
against the previous one, not against this table. A number without a stated CPU
is not comparable to anything, so the host is part of every row.

@@BASELINE_TABLE@@

## Fixtures

`benches/support/` loads only fixtures already checked into the repository:

| Helper | Fixture |
| --- | --- |
| `av1_lossless_intra_stream` / `av1_lossless_intra_frame` | `tests/fixtures/codec/av1_lossless_17x9.hex` |
| `av1_inter_stream` / `av1_inter_temporal_units` | `tests/fixtures/codec/av1_inter_show_existing_16x16.hex` |
| `bundled_hevc_sample` | `examples/media/BigBuckBunny.mp4` |
| `synthetic_yuv420_sequence` | generated; encoder inputs without decoding first |

Every one of them is cached in a `OnceLock`, so the demux and decode cost is paid
once per process rather than once per iteration.

## Throughput

`support::FrameWork` describes the pixel work one iteration performs.
`support::report_throughput` sets `Throughput::Elements(frames)` — criterion then
prints a frames/sec rate — and prints the megapixels each frame carries, which
converts that rate to megapixels per second.

## Profile

`[profile.bench]` repeats `[profile.release]`'s `lto = "fat"` and
`codegen-units = 1`. Cargo's `bench` profile only inherits `release`'s *defaults*,
not the values set in `Cargo.toml`, so without this the numbers would be measured
without the whole-crate optimization that shipped builds get.

## Targets

Benchmarks are native-only. They are declared as an explicit `[[bench]]` target
and criterion is a `cfg(not(target_arch = "wasm32"))` dev-dependency, so the
`wasm32` builds neither resolve nor compile them.

`benches/support/` holds everything that is not the measurement: fixtures and
synthetic inputs in `support`, and the scalar-vs-SIMD axis in `support::isa`
(`bench_across_isas`, the bit-exactness guard, and the per-site override
assertion).
