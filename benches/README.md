# Benchmarks

zvidlib's benchmarks run under [criterion](https://docs.rs/criterion) with
`harness = false`. Every target shares the fixture cache in `benches/support/`,
so each fixture is demuxed, decoded, or encoded once per process and each
iteration measures codec work only.

| Target | Scope |
| --- | --- |
| `codec` | the cross-codec groups: fixture smoke decode, encoder inputs, and the bundled 1080p HEVC sample |
| `av1_decode` | the AV1 software decoder: whole-frame decode and every hot stage, scalar versus SIMD |

## Running

```sh
cargo bench                       # every target's default, fast groups
cargo bench --features simd       # the same groups, recorded under `simd=on`
cargo bench --bench av1_decode    # one target
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

## The AV1 decoder suite (`--bench av1_decode`)

`benches/av1_decode.rs` measures the pure-Rust AV1 software decoder end to end
and per hot stage. Every group in it is a per-ISA group, because the point of
the target is where AV1 decode time goes and how much of it vectorizes.

| Group | Stage |
| --- | --- |
| `av1_decode_frame` | whole-frame decode through `native_av1_video_decoder_factory` |
| `av1_inverse_dct_{4x4,8x8,16x16,32x32,64x64}` | inverse DCT, `src/av1_simd/transforms.rs` |
| `av1_inverse_adst_8x8`, `av1_inverse_flipadst_16x16` | the inverse ADST family |
| `av1_deblock`, `av1_deblock_wide`, `av1_deblock_boundary` | deblocking: narrow filters, the wide 8/14-tap filters, and boundary-dominated planes |
| `av1_cdef`, `av1_wiener`, `av1_self_guided` | CDEF and loop restoration |
| `av1_mc_single`, `av1_mc_compound_average`, `av1_mc_blend_mask` | inter prediction, `src/av1_mc.rs` |
| `av1_intra_paeth`, `av1_intra_smooth`, `av1_intra_directional` | intra prediction, `src/av1_intra_pred.rs` |
| `av1_entropy_symbol` | arithmetic symbol decode, `src/av1_entropy.rs` |

A single `scalar` arm is only meaningful here because `simd::set_override`
covers all three of AV1's independent dispatch sites at once (`av1_simd`,
`av1_mc`, `av1_intra_pred`); pinning any one of them alone would leave the
others vectorized.

`av1_entropy_symbol` is deliberately expected to read the same on every arm. The
range decoder is inherently serial and has no vector path, so it is the Amdahl
ceiling on any whole-frame SIMD win: the number to take from it is its share of
`av1_decode_frame`, not a speedup. Its throughput line counts symbols rather
than pixels, so the harness's "Mpx/s" reads as millions of symbols per second.
There is no separate CDF-adaptation measurement because both AV1 decoders in the
crate require `disable_cdf_update = 1`, so `src/av1_cdf.rs`'s tables are read but
never adapted.

This target replaces the ad-hoc, `#[ignore]`d `tests/av1_simd_bench.rs`: its
input generators are now `support::av1_structured_plane`,
`support::av1_flat_blocks_plane`, and `support::av1_wide_tx_grid`, and its
hand-rolled timing loops are the criterion groups above, so the same
measurements now produce stored baselines. The bit-exactness tests that file sat
next to (`tests/av1_simd_intra.rs`, `src/av1_simd/tests.rs`) are correctness
checks and are unchanged.

On an `x86_64` host with AVX2 the arms are `scalar`, `sse4.1`, and `avx2`
instead. These groups deliberately carry the instruction set in the name rather
than the `simd=on`/`simd=off` build tag: the instruction set *is* the measured
axis here, and both arms always appear in the same run.

Filter to one of them the same way as any other group:

```sh
cargo bench --bench codec -- av1_deblock
cargo bench --bench av1_decode -- 'av1_deblock/scalar'
cargo bench --bench av1_decode -- av1_inverse   # every inverse-transform group
```

Note that the correctness guard below runs for every group in a target
regardless of the filter — it is not a criterion benchmark and criterion's
filter does not reach it — so filtering shortens the *timed* part of a run, not
all of it.

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

## Fixtures

`benches/support/` loads only fixtures already checked into the repository:

| Helper | Fixture |
| --- | --- |
| `av1_lossless_intra_stream` / `av1_lossless_intra_frame` | `tests/fixtures/codec/av1_lossless_17x9.hex` |
| `av1_inter_stream` / `av1_inter_temporal_units` | `tests/fixtures/codec/av1_inter_show_existing_16x16.hex` |
| `bundled_hevc_sample` | `examples/media/BigBuckBunny.mp4` |
| `synthetic_yuv420_sequence` | generated; encoder inputs without decoding first |
| `synthetic_av1_stream` | generated; encoded once by `native_av1_video_encoder_factory` |
| `av1_structured_plane` / `av1_flat_blocks_plane` / `av1_wide_tx_grid` | generated; the AV1 kernel-level inputs |

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

The checked-in AV1 vectors are 17x9 and 16x16 conformance streams, which are
correct but far too small to time a decoder with. `synthetic_av1_stream` is
therefore produced by the crate's own AV1 encoder — the bounded lossless
monochrome subset its decoder implements — at 320x180, once per process and
entirely outside the timed loop.

`benches/support/` holds everything that is not the measurement: fixtures and
synthetic inputs in `support`, and the scalar-vs-SIMD axis in `support::isa`
(`bench_across_isas`, the bit-exactness guard, and the per-site override
assertion).
