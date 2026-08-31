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

## Continuous integration

`.github/workflows/ci.yml` carries a `benchmarks` job.

- **Every pull request** compiles the suite with `cargo bench --no-run`, once per
  `simd` arm. It measures nothing. Bench code that stops compiling is the most
  common way a benchmark suite quietly dies, and this catches it for the price of
  a compile.
- **Pushes to `main`, and manual `workflow_dispatch` runs**, additionally take
  timings. Pull requests are deliberately *not* timed: a shared GitHub runner's
  noise floor is far above any threshold worth enforcing, so gating PRs on it
  would produce false failures until someone disabled the job.

A timed run reduces criterion's estimates to a single `bench-estimates.json`,
uploads it as the `bench-baseline` workflow artifact, and compares it against the
artifact the previous successful `main` run left behind
(`.github/scripts/bench_delta.py`). The per-benchmark delta table goes to the job
summary. Anything more than **15%** slower is flagged; nothing fails the build.
That threshold is deliberately loose and should be tightened once a few weeks of
runs show what the real run-to-run variance on these runners is.

A fresh runner has no `target/criterion` tree, so criterion's own
change-since-last-run detection has nothing to compare against; the stored
artifact is what gives it a previous run at all.

### Instruction sets actually exercised

The crate's vector kernels are chosen by runtime CPU feature detection, so a
benchmark run on a host without AVX2 measures the SSE4.1 or scalar kernel while
saying nothing about it, and GitHub runners vary in AVX2 availability. The suite
therefore prints a host line before any group runs:

```text
# zvidlib benches: arch aarch64, simd feature off, instruction sets available: scalar, neon
```

It comes from `av1_simd::available_isas()` and `av1_mc::available_levels()`, and
appears in the CI log for every timed run. Group names currently distinguish only
the `simd=on` / `simd=off` cargo arm, not the instruction set, so there is no
per-ISA group that could be mislabelled today; when per-ISA groups are added the
absent ones should be skipped rather than recorded under an `avx2` label.

## Baselines

Numbers without a stated CPU are not comparable, so the host is part of the table.

**Host:** Apple M1 (`aarch64`, 8 cores), macOS 26.5.2, stable Rust from
`rust-toolchain.toml`, `cargo bench` (`[profile.bench]`: `lto = "fat"`,
`codegen-units = 1`). Available instruction sets: `scalar`, `neon`.

BASELINE_TABLE_PLACEHOLDER

`simd` gates no code today, so the two arms measure the same kernels; the columns
exist to show the arms are recorded separately and to be the place per-codec SIMD
work lands its numbers. On this host every vector kernel runs NEON, since NEON is
mandatory in the aarch64 base architecture. An `x86_64` row with SSE4.1 and AVX2
columns should be added from a run on such a host.

These are a committed reference point for reading a CI delta report, not a
pass/fail gate. Re-measure them on the same host after a change that is expected
to move them, and state the host if you replace the table.
