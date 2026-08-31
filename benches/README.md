# Benchmarks

zvidlib's benchmarks run under [criterion](https://bheisler.github.io/criterion.rs/book/),
with HTML reports enabled. They are native-only: `wasm32` has no criterion
harness and none of the vector kernels these benchmarks compare, so every bench
target compiles away there.

The point of this harness is **scalar versus SIMD on the same input**. Each
group runs once per instruction set `zvidlib::simd::available()` reports and is
named `<codec>/<isa>`, so criterion compares the arms directly:

```
hevc_decode/scalar
hevc_decode/neon
av1_deblock/scalar
av1_deblock/neon
```

On an `x86_64` host with AVX2 the arms are `scalar`, `sse4.1`, and `avx2`
instead.

## Running

Full run (every bench target, every instruction set):

```sh
cargo bench --features native
```

One bench target:

```sh
cargo bench --features native --bench smoke
```

One codec, or one codec on one instruction set — criterion filters on the
benchmark id, which is exactly the `<codec>/<isa>` name above:

```sh
cargo bench --features native --bench smoke -- hevc_decode
cargo bench --features native --bench smoke -- 'hevc_decode/scalar'
```

Compile without running, which is what CI checks:

```sh
cargo bench --features native --no-run
```

HTML reports land in `target/criterion/report/index.html`.

## Forcing one instruction set

The harness already sweeps every available instruction set, so a filter on the
`<codec>/<isa>` id is usually all you need. To pin the kernels from your own
code — a profiler run, an example, an `#[ignore]`d test — use the public
override:

```rust
use zvidlib::simd::{self, SimdIsa};

simd::set_override(Some(SimdIsa::Scalar)); // every kernel, HEVC and AV1
// ... run the workload ...
simd::set_override(None);                  // back to per-host detection
```

`set_override` reaches all four dispatch families at once: the AV1 transforms
and in-loop filters, AV1 motion compensation, AV1 intra prediction, and every
HEVC engine kernel (inter/intra prediction, in-loop filters, inverse
transforms, and encoder-side distortion metrics). An instruction set this host
cannot execute is clamped to `SimdIsa::Scalar` rather than silently ignored, so
the arm you asked for is always a defined one. `simd::active()` reports what is
in force and `simd::available()` lists what this host can run.

## Build profile

`[profile.bench]` inherits from `[profile.release]`, so benchmarks are built
with `lto = "fat"` and `codegen-units = 1` — the same settings zvidlib ships.
Cargo's built-in `bench` profile does *not* pick up `[profile.release]`
customizations on its own, so without that explicit inherit the numbers would
come from a build nobody uses.

## Writing a new bench target

`benches/support/` holds everything that is not the measurement:

- `support::fixtures` — the bundled `examples/media/BigBuckBunny.mp4` and the
  `tests/fixtures/` AV1 elementary streams, demuxed/parsed **once per process**
  and handed out by reference, so per-iteration cost is codec work only.
- `support::synth` — deterministic YUV420 frame sequences, so encoder-side
  benchmarks have input without decoding something first.
- `support::harness` — `bench_across_isas`, which sweeps the instruction sets,
  names the groups, reports `Throughput::Elements(frames)` plus a megapixels
  per second line, and asserts every arm is bit-exact with scalar before timing
  anything.

That last check is not optional. Every vector backend in the crate is
documented as bit-exact with its scalar reference; a speedup measured on a
kernel that quietly diverged would look like progress. If a backend disagrees
with scalar, the benchmark panics instead of reporting a number.

Declare new targets explicitly in `Cargo.toml` (`autobenches = false` keeps
`benches/support/` from being auto-discovered as a benchmark of its own):

```toml
[[bench]]
name = "my_codec"
path = "benches/my_codec.rs"
harness = false
```
