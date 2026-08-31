# Benchmarks

zvidlib's benchmarks run under [criterion](https://docs.rs/criterion) with
`harness = false`, across five bench targets that share `benches/support/`:

| Target | Measures |
| --- | --- |
| `benches/codec.rs` | codec work: decode, encoder inputs, and the per-ISA SIMD groups |
| `benches/av1_decode.rs` | the AV1 software decoder: whole-frame decode and every hot stage, scalar versus SIMD |
| `benches/audio_decode.rs` | the audio decode path: AAC access units and `AacSampleReader` range/seek reads |
| `benches/audio_mux.rs` | the audio container path: MP4 muxing, sample-table growth, demux, and gapless timing |
| `benches/hevc_encode.rs` | the pure-Rust HEVC encoder, whole-frame and per-stage |

Each target loads and decodes its fixtures once per process, so every iteration
measures the work under test and nothing else. `codec` is one target rather than
several because its groups share the same decoded-frame cache; the AV1 decoder
suite, the two audio targets and `hevc_encode` share none of those fixtures and
are separately runnable. The encoder target is separate for a second reason too:
its mode search is slow enough that keeping it out of the default
`cargo bench --bench codec` run is worth more than sharing a process.

## Running

```sh
cargo bench                       # the default, fast groups in every target
cargo bench --bench codec         # codec work only
cargo bench --bench av1_decode    # the AV1 software decoder only
cargo bench --bench audio_decode  # the audio decode path only
cargo bench --bench audio_mux     # the audio container path only
cargo bench --bench hevc_encode   # the HEVC encoder groups only
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

The same variable gates the encoder target's 1080p-class groups:

```sh
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench hevc_encode
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

## The audio container path

`cargo bench --bench audio_mux` measures the audio write and read paths in two
groups.

`audio_mux` is the write side:

* `media_output_1s_30fps` drives a whole synchronized `MediaOutput` session --
  index checking, timeline interval validation, encoder dispatch, muxer writes,
  the gapless drain, and finalization -- over one second of 48 kHz stereo audio
  and 30 video frames.
* `sample_table_1500_packets` and `sample_table_15000_packets` drive the muxer
  alone over a long audio-only track. The muxer writes one chunk per sample, so
  `stsz` and `co64` each grow by a fixed width per sample while `stts` and `stsc`
  stay run-length constant. The two sizes are an order of magnitude apart on
  purpose: the *ratio* between them is the regression guard, and it should stay
  close to linear.

`audio_demux` is the read side, over the bundled sample's real AAC track:
`Mp4Demuxer::open` (which parses those same sample tables back),
`to_encoded_audio_samples` (packet extraction over the decoded sample clock), and
`audio_timing` (priming, padding, and edit-list mapping).

### There is no audio encoder to benchmark

`AudioEncoder` (`src/codec.rs`) is a trait with no implementation in the crate.
Its only implementor anywhere in the tree is `PcmFixtureEncoder` in
`tests/indexed_mp4_output.rs`, a test double that packages PCM without
compressing anything. "Benchmark the audio encoder" therefore has no subject, and
this target does not invent one -- whether the crate should grow a native AAC-LC
encoder, delegate to a platform encoder (AudioToolbox / Media Foundation), or
leave the trait for platform and web backends to fill is a product decision.

The bench-local `PcmBenchEncoder` is the same kind of pass-through double, and it
is bench-local on purpose: holding codec work at effectively zero is what makes
the measurement isolate container work.

### No SIMD axis

Muxing and demuxing are bit-shuffling and table building, not arithmetic over
sample arrays, so there is nothing here for a vector kernel to do. These groups
run on the detected instruction set only and do not use `bench_across_isas`. They
still carry the `simd=off` / `simd=on` build tag every group name carries,
because that tag records which *build* produced a number, not which kernel ran.

## Fixtures

`benches/support/` loads only fixtures already checked into the repository:

| Helper | Fixture |
| --- | --- |
| `av1_lossless_intra_stream` / `av1_lossless_intra_frame` | `tests/fixtures/codec/av1_lossless_17x9.hex` |
| `av1_inter_stream` / `av1_inter_temporal_units` | `tests/fixtures/codec/av1_inter_show_existing_16x16.hex` |
| `bundled_hevc_sample` | `examples/media/BigBuckBunny.mp4` |
| `bundled_aac_track` / `bundled_mp4_bytes` | `examples/media/BigBuckBunny.mp4` (its AAC-LC stereo track) |
| `aac_mono_track` | `tests/fixtures/codec/aac_lc_mono_48k.m4a` |
| `synthetic_yuv420_sequence` | generated; encoder-stage inputs without decoding first |
| `synthetic_rgba8_sequence` | generated; whole-frame encoder inputs (the public encoder takes RGBA8) |
| `synthetic_av1_stream` | generated; encoded once by `native_av1_video_encoder_factory` |
| `av1_structured_plane` / `av1_flat_blocks_plane` / `av1_wide_tx_grid` | generated; the AV1 kernel-level inputs |

Every one of them is cached in a `OnceLock`, so the demux and decode cost is paid
once per process rather than once per iteration.

## Throughput

`support::FrameWork` describes the pixel work one iteration performs.
`support::report_throughput` sets `Throughput::Elements(frames)` — criterion then
prints a frames/sec rate — and prints the megapixels each frame carries, which
converts that rate to megapixels per second.

Audio has no pixels, so it reports on its own scale: `support::AudioWork` and
`support::report_audio_throughput` set `Throughput::Elements(samples)` — a
per-channel samples/sec rate — and print the sample rate and covered duration,
which convert that rate to a factor of realtime. Both sides of the write path and
both sides of the read path use it, so mux, demux, and AAC decode numbers stay
directly comparable. A benchmark that touches no samples (`audio_timing`, which is
O(edits)) deliberately registers no throughput rather than reporting a fabricated
sample rate.

The `audio_decode` groups additionally print their own `NNNx realtime` line from a
single separately timed pass, because x-realtime is the figure a playback path is
judged by and criterion has no unit for it.

## The HEVC encoder target

`benches/hevc_encode.rs` measures the pure-Rust HEVC encoder on two axes.

**Whole-frame** groups encode a fixed-length synthetic RGBA8 sequence through
the public `native_hevc_video_encoder_factory` and report frames/sec and
megapixels/sec. Nothing is decoded first, so no decoder cost is folded into an
encoder number.

**Per-stage** groups time the pipeline's stages individually, so the mode-search
cost — which dominates everything else by an order of magnitude — is not
mistaken for bitstream-writing cost:

| Group | Stage |
| --- | --- |
| `..._rdo_intra` / `..._rdo_inter` | mode search / RDO (`engine::encoder::rdo`), without and with a reference picture |
| `..._pcm_write` | whole-picture access-unit writing: parameter sets, slice header, CABAC-coded CU syntax, PCM samples |
| `hevc_encode_cabac` | the §9.3.5 arithmetic encoder alone, over a synthetic bin stream |
| `hevc_encode_bitwriter` | the raw fixed-length / `ue(v)` / `se(v)` writer alone |
| `..._rgba_to_yuv420` | the RGBA8 input conversion every encoded frame pays |

Every group runs once per available instruction set through
`support::isa::bench_across_isas`, with the same bit-exactness and
`active_by_site()` guards as the decode-side groups.

### Resolutions

640x352 always; 1920x1088 behind `ZVIDLIB_BENCH_LARGE=1`. The PCM writer
requires dimensions divisible by 16, so the nominal 640x360 and 1920x1080 are
not encodable — these are the nearest valid sizes at the same scale.

### Stages this encoder does not have yet

The encoder is a lossless PCM bootstrap writer. It has **no forward transform,
no quantization, and no encoder-side reconstruction or in-loop filtering**: PCM
samples are written verbatim, so there is no residual to transform and no
reconstructed picture that could differ from the source. Those stages have no
group here because they do not exist yet, and the target prints that on every
run so a missing group is never read as a stage that costs nothing.

### Where the SIMD axis reads flat, and why that is the result

`hevc_rdcost` — the SAD and SATD distortion metrics the mode search calls — is
the encoder's **only** SIMD dispatch family. Bitstream writing, CABAC, and the
RGBA-to-YUV420 conversion have no vector path at all, so their arms are expected
to read the same under every instruction set. That is a measured result, not a
broken benchmark: it says the next encoder-side vectorization targets are
entropy coding and color conversion. It is also why every group asserts through
`simd::active_by_site()` that the override landed rather than inferring it from
the clock — see [Reading a null result](#reading-a-null-result).

## Per-stage access to the encoder

`crate::hevc` is a private module and benchmarks are a separate crate, so the
encoder's stages are not reachable through the public API — the public factory
runs all of them at once, which is exactly what a per-stage breakdown must
avoid. `zvidlib::hevc_encoder_bench` is that access: `#[doc(hidden)]`, explicitly
not part of the stable API, and nothing but thin wrappers that own their inputs
and return the bytes identifying their result. Returning those bytes is what
keeps the bit-exactness guard armed; a stage whose return value did not depend
on the kernels under test would disarm it silently.

## Profile

`[profile.bench]` repeats `[profile.release]`'s `lto = "fat"` and
`codegen-units = 1`. Cargo's `bench` profile only inherits `release`'s *defaults*,
not the values set in `Cargo.toml`, so without this the numbers would be measured
without the whole-crate optimization that shipped builds get.

## Targets

Benchmarks are native-only. They are declared as explicit `[[bench]]` targets and
criterion is a `cfg(not(target_arch = "wasm32"))` dev-dependency, so the `wasm32`
builds neither resolve nor compile them.

Each bench target is its own crate root and compiles all of `benches/support/`,
but uses only the fixtures its own measurements need, so the module carries
`#![allow(dead_code)]`: without it `cargo clippy --all-targets` would fail one
target over helpers another target depends on.

The checked-in AV1 vectors are 17x9 and 16x16 conformance streams, which are
correct but far too small to time a decoder with. `synthetic_av1_stream` is
therefore produced by the crate's own AV1 encoder — the bounded lossless
monochrome subset its decoder implements — at 320x180, once per process and
entirely outside the timed loop.

`benches/support/` holds everything that is not the measurement: fixtures and
synthetic inputs in `support`, and the scalar-vs-SIMD axis in `support::isa`
(`bench_across_isas`, the bit-exactness guard, and the per-site override
assertion).

## Audio groups

`benches/audio_decode.rs` measures two layers, and keeps them in separate groups
on purpose:

| Group | What it measures |
| --- | --- |
| `aac_decode` | `NativeAacDecoder::decode` over a fixed run of access units, mono and stereo |
| `aac_reader_sequential` | `AacSampleReader::get_range` re-reading a resident range, and walking forward |
| `aac_reader_seek` | random-access ranges, each forcing a decoder reset and a preroll re-decode |
| `aac_reader_edits` | reads crossing edit-list boundaries and gapless priming/padding trims |

`AacSampleReader` keeps decoded packets in a `BTreeMap`, so the same call costs
two very different things depending on whether the requested media range is
already resident. `cached_repeat_stereo_48k` is the pure hit path — no decode at
all — and everything in `aac_reader_seek` is the cold path. Reporting them
together would average the seek cost away, and the seek cost is the one that
shows up as an audible stall.

These groups carry **no `simd=` tag and no per-ISA arms**. AAC decoding is
delegated to the third-party `symphonia-codec-aac` crate, `zvidlib::simd`'s
override does not reach it, and the crate has no audio SIMD kernels of its own,
so a scalar arm and a vector arm would be the same code reported twice.

The mono fixture exists because the bundled sample is stereo and carries no edit
list, while `NativeAacDecoder` accepts AAC-LC mono as well (and rejects
everything beyond stereo, so mono and stereo are its entire supported input
space). It also supplies the real `elst` and decoder-priming timing the bundled
sample does not have.

Every bench target shares `benches/support/`, so each one leaves some of its
helpers unused; the module allows `dead_code` for that reason.
