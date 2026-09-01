# Benchmarks

zvidlib's benchmarks run under [criterion](https://docs.rs/criterion) with
`harness = false`, across six bench targets that share `benches/support/`:

| Target | Measures |
| --- | --- |
| `benches/codec.rs` | codec work: decode, encoder inputs, and the per-ISA SIMD groups |
| `benches/av1_decode.rs` | the AV1 software decoder: whole-frame decode and every hot stage, scalar versus SIMD |
| `benches/av1_encode.rs` | the native AV1 encoder: whole-frame encode, every stage, and the forward-transform kernels, scalar versus SIMD |
| `benches/audio_decode.rs` | the audio decode path: AAC access units and `AacSampleReader` range/seek reads |
| `benches/audio_mux.rs` | the audio container path: MP4 muxing, sample-table growth, demux, and gapless timing |
| `benches/hevc_encode.rs` | the pure-Rust HEVC encoder, whole-frame and per-stage |

Each target loads and decodes its fixtures once per process, so every iteration
measures the work under test and nothing else. `codec` is one target rather than
several because its groups share the same decoded-frame cache; the two AV1
suites, the two audio targets and `hevc_encode` share none of those fixtures and
are separately runnable. The AV1 decode and encode suites are separate targets
from each other so neither one's name overstates what it measures: encoder
kernels are not decoder stages, even where both reach the same `av1_simd`
dispatch site. The encoder target is separate for a second reason too:
its mode search is slow enough that keeping it out of the default
`cargo bench --bench codec` run is worth more than sharing a process.

## Running

```sh
cargo bench                       # the default, fast groups in every target
cargo bench --bench codec         # codec work only
cargo bench --bench av1_decode    # the AV1 software decoder only
cargo bench --bench av1_encode    # the AV1 encoder, whole-frame and per-stage
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

The forward transforms are not in this table. They are encoder kernels and are
measured by [the AV1 encoder suite](#the-av1-encoder-suite---bench-av1_encode)
instead, even though they reach the same `av1_simd` dispatch site as the inverse
transforms above.

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

## The AV1 encoder suite (`--bench av1_encode`)

`benches/av1_encode.rs` measures the native AV1 encoder on two axes: whole-frame
versus per-stage, and instruction set.

The whole-frame groups encode a synthetic monochrome frame through the public
`zvidlib::native_av1_video_encoder_factory` at three quantizer settings and
report frames/sec and megapixels/sec. The per-stage groups run each stage on its
own through `zvidlib::av1_encoder_bench`, the `#[doc(hidden)]` per-stage access
that is the AV1 counterpart to `hevc_encoder_bench`, so the tile encoder's cost
is not mistaken for bitstream-writing cost. Both default to 640x352 and add a
1920x1080 pass behind `ZVIDLIB_BENCH_LARGE=1`:

| Group | Stage |
| --- | --- |
| `av1_encode_frame_q{0,32,160}` | one whole frame through the public encoder, `src/av1_encoder/tile.rs` |
| `av1_encode_stage_wht` | the forward 4x4 WHT, `src/av1_encoder/wht.rs` |
| `av1_encode_stage_symbol` | symbol coding over the static CDF tables, `src/av1_encoder/symbol.rs` and `cdf.rs` |
| `av1_encode_stage_tile` | tile encoding: superblock iteration, `DC_PRED`, coefficient coding, `src/av1_encoder/tile.rs` |
| `av1_encode_stage_bitstream` | headers, bit writing and OBU LEB128 framing, `src/av1_encoder/{bitwriter,headers,leb128}.rs` |
| `av1_forward_dct_{4x4,8x8,16x16,32x32}` | forward DCT, `src/av1_encoder/transform.rs` through `zvidlib::forward_transform` |
| `av1_forward_adst_8x8`, `av1_forward_flipadst_16x16` | the forward ADST family, including a flipped type |

The per-stage groups run at `base_q_idx = 0`, the lossless WHT profile, so they
decompose the same work `av1_encode_frame_q0` measures end to end. The
non-lossless search the `q32` and `q160` groups show is a whole-frame property
rather than a stage of its own, which is why it has no per-stage counterpart.

The stage breakdown is the point of the per-stage groups, and it is lopsided:
tile encoding is within a small factor of the whole-frame number, the forward
WHT and the symbol coder are each an order of magnitude cheaper than that, and
header writing with its LEB128 framing is two to three orders cheaper again —
microseconds against a 1080p tile encode's hundreds of milliseconds. Coefficient
coding and its context derivation inside `tile.rs`, not the transform, are what a
faster lossless encoder has to attack next.

The forward transforms and the forward WHT are this encoder's only vectorized
kernels. Symbol coding, CDF handling and bitstream writing are scalar and
expected to stay that way, so those arms read the same under every instruction
set. That flatness is a measured result rather than a broken run, which is why
each group asserts through `simd::active_by_site()` that the override landed
instead of inferring it from the clock. `report_stage_coverage` prints the stage
list on every run, so a group that stops being measured reads as a broken run
rather than as a stage that costs nothing.

```sh
cargo bench --bench av1_encode -- av1_encode_stage    # the per-stage groups only
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench av1_encode  # add the 1080p pass
```

The kernel-level `av1_forward_*` groups run once per available instruction set
through
`support::isa::bench_across_isas`, under the same bit-exactness and
`active_by_site()` guards as every other per-ISA group, and over the same
1920x1080 block counts and coefficient generator as the inverse-transform
groups in `av1_decode` — so the two directions stay directly comparable:

```sh
cargo bench --bench av1_encode -- av1_forward
cargo bench --bench av1_encode -- 'av1_forward_dct_16x16/scalar'
```

Note that the correctness guard below runs for every group in a target
regardless of the filter — it is not a criterion benchmark and criterion's
filter does not reach it — so filtering shortens the *timed* part of a run, not
all of it.

The whole-frame per-ISA HEVC groups (`hevc_decode` and
`hevc_decode_to_picture`) decode the bundled 1080p sample, so they sit behind
the same `ZVIDLIB_BENCH_LARGE=1` opt-in as the other 1080p group.

### Which whole-frame HEVC group answers which question

There are two, and they measure deliberately different intervals:

| Group | Interval | The question it answers |
| --- | --- | --- |
| `hevc_decode/<isa>` | `submit` to RGBA | what does an application pay per frame |
| `hevc_decode_to_picture/<isa>` | `submit` to the decoded `Picture` | how fast is the decoder |

They decode the same access units of the same sample, the same number of frames,
through the same `HevcDecoder`. The only difference is that the first converts
each decoded picture to RGBA and the second does not.

The distinction exists because that conversion is **33.5% of what `hevc_decode`
times** (the attribution below) and no HEVC kernel touches it. A scalar-versus-SIMD
ratio taken off `hevc_decode` therefore has a third of its denominator pinned
regardless of how fast the vector kernels get: it is not a decode ratio, and
reading it as one understates every SIMD arm. `hevc_decode_to_picture` is the
group to read a decode ratio off. `hevc_decode` stays because the round trip is
what an application actually pays, and dropping it would trade one misleading
number for another.

Both arms fold their output with the same cheap FNV step for the bit-exactness
guard — over the RGBA bytes in one case and the picture's planes in the other —
so the gap between the groups is the conversion and not an artefact of how each
identifies its result. (`hevc_decode` previously took a `FrameDigest` per frame
inside the timed loop; SHA-256 over an 8 MB frame cost more than the decode it
was measuring, which inflated the group and buried the conversion it was meant
to expose.) `hevc_color_convert` below still times the conversion directly, and
that is the number to quote for it.

`hevc_decode_1080p` reports both intervals the same way:
`sequential_from_keyframe` goes through `ExactFrameReader` out to RGBA, and
`sequential_from_keyframe_to_picture` decodes the same leading frames and stops
at the picture.

### The HEVC per-stage groups

The whole-frame groups answer "how fast is a frame". The per-stage groups answer
"which kernel changed", so a regression can be attributed rather than only
observed:

| Group | Stage | Vectorized |
| --- | --- | --- |
| `hevc_inter_pred` | §8.5.3.3 8-tap luma interpolation + the weighted combine | yes |
| `hevc_intra_pred` | §8.4.4.2 reference smoothing, planar / DC / angular | yes |
| `hevc_deblock` | §8.7.2 luma block-edge deblocking | yes |
| `hevc_sao` | §8.7.3 sample adaptive offset, band and edge | yes |
| `hevc_inverse_transform` | §8.6 dequantization + inverse DCT/DST | yes |
| `hevc_color_convert` | decoder output YUV420-to-RGBA conversion | yes |
| `hevc_cabac` | §9.3.4 arithmetic bin decoding | no, by design |

They run unconditionally — none of them touches the bundled sample, so none of
them needs the `ZVIDLIB_BENCH_LARGE=1` opt-in — and each runs once per available
instruction set under the same bit-exactness and per-site override guards as
every other per-ISA group.

`hevc_color_convert` is the odd one out in the other direction: it is not a
decoding stage at all, but the fixed BT.601/709 integer YUV-to-RGBA pass every
decoded picture takes on the way out of the decoder. It is here because the
breakdown below measured it as the single largest item in a whole-frame decode,
and because it is on the path of *both* whole-frame groups — so it is the one
per-stage group whose arms directly explain part of the `hevc_decode/<isa>`
ratio rather than only bounding it.

`hevc_cabac` is in the list precisely because it is *not* vectorized. The
arithmetic decoder is inherently serial (each bin's range update depends on the
previous one's), so whatever fraction of a decode it owns is the fraction no
amount of SIMD elsewhere can remove. Its arms should come out equal; the number
is meaningful next to the other stages, as the Amdahl ceiling on the whole-frame
group. Its throughput axis counts bins, so its `Mpx/s` line reads as
megabins/sec.

The inputs come from `zvidlib::hevc_decoder_bench`, a narrow public surface over
the otherwise crate-private HEVC engine. Its `HevcStageInputs::new` does all the
allocation and content generation; only the kernel under test runs inside
`run_*`. The content deliberately mixes textured and flat regions: the wide
deblocking filter is gated on the §8.7.2.5.3 flatness check, so a purely
textured plane would time only the narrow path and under-report the kernel.

Each `run_*` returns an eight-byte FNV-1a fold over every sample the stage
produced rather than the samples themselves, which keeps a multi-megabyte
allocation out of the timed loop while still letting the bit-exactness guard
catch a backend that diverged anywhere.

### Where HEVC decode time actually goes

The per-stage groups above time each kernel on a workload of its own, which
bounds what vectorizing that stage *could* buy. It does not say what it *does*
buy, because it does not say how much of a real frame goes through the stage.
Issue #189 was that gap: `hevc_decode/<isa>` moved only ~1.06x between the
`scalar` and `neon` arms while §8.5.3.3 luma interpolation measures 1.6-1.7x,
§8.7.3 SAO 2.4x and §8.7.2 deblocking 1.3x in isolation. The breakdown below is
what identified the missing colour-conversion kernel (#219) as the largest
single reason.

`examples/hevc_decode_profile.rs` closes it. It decodes the bundled sample
through the ordinary public decoder with `zvidlib::hevc_decode_profile` running,
which charges wall time to a stage *exclusively* — the §7.3.8.11 residual parse
nested inside the §7.3.8 slice-data walk is subtracted from it rather than
counted twice — and prints each stage's share:

```sh
cargo run --release --features native --example hevc_decode_profile
cargo run --release --features native --example hevc_decode_profile -- 120
cargo run --release --features native --example hevc_decode_profile -- 48 scalar
```

The second positional argument is a frame count and the third pins an
instruction set. `native` gates no code in the example; it is what keeps the
wasm build, which has no HEVC decoder, from compiling a native-only target.

It is an example rather than a criterion group because the question is how one
decode divides, which is a composition rather than a number to regress against.
Run it under `--release`: at `opt-level = 2` the stage mix is not the shipped
build's.

#### The breakdown

48 frames of `examples/media/BigBuckBunny.mp4` (1920x1080, HEVC Main 8-bit
4:2:0) on an Apple Silicon host, `--release`, best of three interleaved rounds
per arm. Both arms are reported side by side because since issue #219 gave
`color_convert` a kernel of its own they no longer agree to within a point: the
`scalar` arm runs 30.20 ms/frame and the `neon` arm 21.39 ms/frame, a 1.41x
whole-frame ratio where the pre-#219 measurement read ~1.06x.

`scalar` arm, 30.20 ms/frame:

| Stage | Share of total | Share of decode | ms/frame | Vectorized |
| --- | ---: | ---: | ---: | --- |
| `color_convert` | 32.7% | n/a | 9.86 | yes |
| `inter_pred` | 23.1% | 34.3% | 6.98 | yes |
| `sao` | 9.2% | 13.7% | 2.78 | yes |
| `deblock` | 8.0% | 11.9% | 2.43 | yes |
| `intra_pred` | 3.9% | 5.7% | 1.16 | yes |
| `residual_cabac` | 3.7% | 5.5% | 1.12 | no |
| `motion_derive` | 3.3% | 4.9% | 1.00 | no |
| `inverse_transform` | 2.9% | 4.2% | 0.86 | yes |
| `slice_data_cabac` | 2.1% | 3.1% | 0.62 | no |
| `dpb_output` | 1.8% | 2.6% | 0.53 | no |
| `header_parse` | 0.0% | 0.0% | 0.01 | no |
| _unattributed_ | 9.4% | 14.0% | 2.85 | n/a |

`neon` arm, 21.39 ms/frame:

| Stage | Share of total | Share of decode | ms/frame | Vectorized |
| --- | ---: | ---: | ---: | --- |
| `inter_pred` | 29.6% | 32.6% | 6.33 | yes |
| `sao` | 13.0% | 14.3% | 2.77 | yes |
| `deblock` | 10.6% | 11.7% | 2.27 | yes |
| `color_convert` | 9.3% | n/a | 1.98 | yes |
| `residual_cabac` | 5.3% | 5.8% | 1.12 | no |
| `intra_pred` | 5.2% | 5.7% | 1.12 | yes |
| `inverse_transform` | 5.2% | 5.7% | 1.10 | yes |
| `motion_derive` | 4.4% | 4.8% | 0.93 | no |
| `slice_data_cabac` | 2.8% | 3.1% | 0.61 | no |
| `dpb_output` | 2.2% | 2.4% | 0.47 | no |
| `header_parse` | 0.0% | 0.0% | 0.00 | no |
| _unattributed_ | 12.5% | 13.8% | 2.68 | n/a |

"Share of decode" divides by the total minus `color_convert`, because colour
conversion is not decoding: it is the YUV420-to-RGBA pass every whole-frame
measurement takes on the way out of the decoder. Both denominators are reported
because the two answer different questions and are easy to confuse.

_unattributed_ is real work in no instrumented scope — the coding-quadtree and
CTU walks, the per-CU residual extraction glue, and allocation between stages.
It is left as its own row rather than spread across the stages, so no share is
inflated by work it does not do. The profiler's own cost is under 1% of the
total at ~327k scopes; the example prints that bound on every run.

#### What it says

**The issue's hypothesis was wrong.** Entropy decoding is not where the time
goes. `slice_data_cabac` and `residual_cabac` together are **5.8% of the total
and 8.6% of decode proper** — the §9.3.4 arithmetic decoder is serial and has no
vector path, but it is nowhere near large enough to be the reason whole-frame
SIMD reads flat.

**Vectorized stages cover 79.7% of the measured total and 69.9% of decode
proper.** By Amdahl, infinitely fast vector kernels would move the measured
whole-frame number 4.93x, a uniform 2x on those stages gives 1.66x, and a
uniform 4x gives 2.49x. So the kernels are *not* a minority of decode time — the
ceiling is high enough that the observed whole-frame ratio is a shortfall
against it, not a consequence of it.

**The largest item was not decoding at all, and it now has a kernel.**
`color_convert` — the per-sample BT.601/709 integer conversion in
`picture_to_rgba` — was a third of everything the whole-frame groups measure
with no vector path whatsoever, so every whole-frame SIMD number was diluted by
roughly a third for a stage no HEVC kernel touched. Issue #219 vectorized it
(`src/hevc/color_convert.rs`, timed by the `hevc_color_convert` group), and it
falls from **9.86 ms/frame to 1.98 ms/frame — 5.0x** — which is most of why the
whole-frame ratio moved from ~1.06x to 1.41x on this host. It is still a third
of the `scalar` arm, because that arm is what a *scalar* colour conversion
costs; on the `neon` arm it is 9.3%.

**The next target is `inter_pred`.** At 32.6% of decode proper on the `neon`
arm it is now comfortably the largest stage, and it is already vectorized — so
the work there is the #166 / #202 question of why its measured arms sit closer
to parity on this host than its isolated kernel numbers suggest, rather than a
question of coverage. Entropy decoding is still not the answer: `slice_data_cabac`
and `residual_cabac` together are 5.8% of the `scalar` total.

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
compressing anything. "Benchmark the audio encoder" therefore has no subject.

That question is now closed rather than open. zvidlib ships no audio encoder by
decision: the trait is the seam that platform and browser backends fill, and the
rationale is recorded on `AudioEncoder` in `src/codec.rs` and in the README. So
there is no audio-encode target pending for this suite, and none of the audio
groups below are placeholders waiting on one.

The bench-local `PcmBenchEncoder` is the same kind of pass-through double, and it
is bench-local on purpose: holding codec work at effectively zero is what makes
the measurement isolate container work.

### No SIMD axis

Muxing and demuxing are bit-shuffling and table building, not arithmetic over
sample arrays, so there is nothing here for a vector kernel to do. These groups
run on the detected instruction set only and do not use `bench_across_isas`. They
still carry the `simd=off` / `simd=on` build tag every group name carries,
because that tag records which *build* produced a number, not which kernel ran.

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

15% is deliberately loose and deliberately provisional. A threshold guessed
tighter than the noise floor produces false regressions immediately — the same
failure mode as gating PRs on timings, just slower — so it was picked wide
enough to be quiet until there was something to calibrate it against.

**It has not been calibrated yet, because the data does not exist yet.** The
timed job that stores baselines landed with the delta report itself; calibrating
it needs a run of consecutive `main` pushes measured through that job, and those
accumulate at the rate `main` moves. Until then 15% stands as a guess that
nobody has checked, not as a number the suite's measured spread supports. Do not
quote it as if it were the latter, and do not tighten it on a hunch: a threshold
moved without data is the same guess at a different value.

The comparison uses criterion's **median** point estimate rather than the mean,
because one descheduled iteration on a shared runner moves the mean and leaves
the median alone. It compares point estimates rather than running criterion's
own change detection, which needs both runs' raw sample data in one
`target/criterion/` directory and assumes the same machine produced both.

### Calibrating it

The measurement is a command rather than a project. `criterion_baseline.py
variance` takes the stored baselines in chronological order and reports, per
group, how far a benchmark moves between two runs when nothing about it
changed — the same `|median|` delta the report thresholds, over pairs where the
code did not change meaningfully.

```sh
# Every stored main baseline, oldest first. They expire after 90 days.
n=0
gh api 'repos/lsegal/zvidlib/actions/artifacts?name=criterion-baseline-main&per_page=100' \
  --jq '[.artifacts[] | select(.expired == false)] | reverse | .[] | [.id, .workflow_run.head_sha] | @tsv' \
  | while IFS=$'\t' read -r id sha; do
      # Numbered, not named after the commit: `variance` reads chronological
      # order off the argument order, and a `$sha` glob sorts alphabetically.
      n=$((n + 1))
      dir="$(printf 'run-%03d-%s' "$n" "$sha")"
      gh api "repos/lsegal/zvidlib/actions/artifacts/$id/zip" > "$id.zip"
      unzip -o -j "$id.zip" -d "$dir"
    done

python3 .github/scripts/criterion_baseline.py variance \
  --baseline run-*/criterion-baseline.json --out variance.md
```

Per group and not one number for the suite, because a whole-frame 1080p group
and a microbenchmark do not share a noise floor and a single global threshold
may be the wrong shape for both. The report's suggested threshold is the
smallest whole 5% step above the worst delta in the sample: a floor on a
defensible number, not a recommendation, since a sample that happened to miss a
bad run suggests a threshold the next bad run will cross. Below ten pairs the
report marks itself provisional and its p95 column should be ignored — with a
handful of samples the p95 is just the worst thing seen so far.

Reading a tighter threshold off that report is the point of collecting it. A
per-group threshold is a legitimate outcome. So is recording that 15% survived
contact with the data; what is not an outcome is leaving this section saying the
same thing in a year.

### Gating

`--fail-on-regression` exists in `criterion_baseline.py` and is deliberately
unused. Gating a group requires more than a threshold: the group's whole
observed spread has to fit under the gate, and its arms have to be present in
every run — a group whose `avx2` arm comes and goes with the runner pool cannot
be gated on at any threshold, because the disappearance is not a percentage.
`variance` reports both, and neither question can be answered before the
baselines above exist.

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

The table is generated, not hand-typed: `criterion_baseline.py table` renders it
from the same baseline JSON the CI job collects, so refreshing it is a
measurement rather than an edit.

```sh
for round in 1 2 3; do
  for target in codec av1_decode av1_encode hevc_decode hevc_encode; do
    cargo bench --features native --bench "$target"
  done
  python3 .github/scripts/criterion_baseline.py collect \
    --criterion-dir target/criterion --out "baseline-$round.json"
done
python3 .github/scripts/criterion_baseline.py table \
  --baseline baseline-*.json --host 'Apple M1 (macOS 15, aarch64)'
```

Three rounds and not one, because `table` takes the elementwise **minimum**
across the baselines it is given. Contention only ever makes a measurement
slower, so the fastest observation of an arm is the closest any round got to an
uncontended one; averaging would fold every neighbour process into the number
instead.

Only the groups `bench_across_isas` builds appear, because they are the only
ones where "scalar vs each ISA" is a question — the rest of the suite is a single
arm with nothing to compare against.

Measured on **Apple M1 (macOS 15, aarch64)**, at `b6655bad215f`.

| Group | `scalar` | `neon` | Best |
| --- | ---: | ---: | ---: |
| `av1_cdef` | 46.615 ms | 32.865 ms (1.42x) | 1.42x `neon` |
| `av1_deblock` | 25.088 ms | 4.497 ms (5.58x) | 5.58x `neon` |
| `av1_deblock_boundary` | 362.401 µs | 62.127 µs (5.83x) | 5.83x `neon` |
| `av1_deblock_chroma` | 12.448 ms | 4.838 ms (2.57x) | 2.57x `neon` |
| `av1_deblock_wide` | 78.324 ms | 40.843 ms (1.92x) | 1.92x `neon` |
| `av1_decode_frame` | 82.680 ms | 88.976 ms (0.93x) | 0.93x `neon` |
| `av1_encode_frame_q0` | 22.174 ms | 24.314 ms (0.91x) | 0.91x `neon` |
| `av1_encode_frame_q160` | 344.933 ms | 153.918 ms (2.24x) | 2.24x `neon` |
| `av1_encode_frame_q32` | 275.371 ms | 170.418 ms (1.62x) | 1.62x `neon` |
| `av1_encode_stage_bitstream` | 17.426 µs | 17.517 µs (0.99x) | 0.99x `neon` |
| `av1_encode_stage_symbol` | 1.558 ms | 1.940 ms (0.80x) | 0.80x `neon` |
| `av1_encode_stage_tile` | 30.144 ms | 31.441 ms (0.96x) | 0.96x `neon` |
| `av1_encode_stage_wht` | 981.525 µs | 360.850 µs (2.72x) | 2.72x `neon` |
| `av1_entropy_symbol` | 3.449 ms | 3.422 ms (1.01x) | 1.01x `neon` |
| `av1_forward_adst_8x8` | 33.915 ms | 8.722 ms (3.89x) | 3.89x `neon` |
| `av1_forward_dct_16x16` | 45.538 ms | 12.369 ms (3.68x) | 3.68x `neon` |
| `av1_forward_dct_32x32` | 60.160 ms | 63.603 ms (0.95x) | 0.95x `neon` |
| `av1_forward_dct_4x4` | 50.984 ms | 7.889 ms (6.46x) | 6.46x `neon` |
| `av1_forward_dct_8x8` | 33.702 ms | 8.557 ms (3.94x) | 3.94x `neon` |
| `av1_forward_flipadst_16x16` | 36.556 ms | 12.258 ms (2.98x) | 2.98x `neon` |
| `av1_intra_directional` | 26.145 ms | 27.222 ms (0.96x) | 0.96x `neon` |
| `av1_intra_paeth` | 3.376 ms | 3.442 ms (0.98x) | 0.98x `neon` |
| `av1_intra_smooth` | 3.564 ms | 3.410 ms (1.05x) | 1.05x `neon` |
| `av1_inverse_adst_8x8` | 50.591 ms | 17.851 ms (2.83x) | 2.83x `neon` |
| `av1_inverse_dct_16x16` | 24.143 ms | 11.105 ms (2.17x) | 2.17x `neon` |
| `av1_inverse_dct_32x32` | 17.398 ms | 11.327 ms (1.54x) | 1.54x `neon` |
| `av1_inverse_dct_4x4` | 78.638 ms | 28.663 ms (2.74x) | 2.74x `neon` |
| `av1_inverse_dct_64x64` | 27.862 ms | 13.821 ms (2.02x) | 2.02x `neon` |
| `av1_inverse_dct_8x8` | 42.501 ms | 17.817 ms (2.39x) | 2.39x `neon` |
| `av1_inverse_flipadst_16x16` | 28.316 ms | 12.777 ms (2.22x) | 2.22x `neon` |
| `av1_mc_blend_mask` | 26.063 ms | 12.135 ms (2.15x) | 2.15x `neon` |
| `av1_mc_compound_average` | 25.797 ms | 11.882 ms (2.17x) | 2.17x `neon` |
| `av1_mc_single` | 16.269 ms | 5.961 ms (2.73x) | 2.73x `neon` |
| `av1_motion_compensation` | 14.660 ms | 5.442 ms (2.69x) | 2.69x `neon` |
| `av1_self_guided` | 8.473 ms | 3.310 ms (2.56x) | 2.56x `neon` |
| `av1_wiener` | 9.564 ms | 7.196 ms (1.33x) | 1.33x `neon` |
| `hevc_cabac` | 2.559 ms | 2.250 ms (1.14x) | 1.14x `neon` |
| `hevc_color_convert` | 15.095 ms | 11.926 ms (1.27x) | 1.27x `neon` |
| `hevc_deblock` | 15.124 ms | 14.748 ms (1.03x) | 1.03x `neon` |
| `hevc_encode_640x352` | 104.222 ms | 57.448 ms (1.81x) | 1.81x `neon` |
| `hevc_encode_640x352_fwd_transform_quant` | 13.874 ms | 7.306 ms (1.90x) | 1.90x `neon` |
| `hevc_encode_640x352_pcm_write` | 10.279 ms | 10.003 ms (1.03x) | 1.03x `neon` |
| `hevc_encode_640x352_rdo_inter` | 77.838 ms | 27.887 ms (2.79x) | 2.79x `neon` |
| `hevc_encode_640x352_rdo_intra` | 3.631 ms | 1.554 ms (2.34x) | 2.34x `neon` |
| `hevc_encode_640x352_reconstruct` | 13.469 ms | 13.733 ms (0.98x) | 0.98x `neon` |
| `hevc_encode_640x352_residual_write` | 34.563 ms | 33.620 ms (1.03x) | 1.03x `neon` |
| `hevc_encode_640x352_rgba_to_yuv420` | 628.049 µs | 151.822 µs (4.14x) | 4.14x `neon` |
| `hevc_encode_bitwriter` | 4.208 ms | 4.060 ms (1.04x) | 1.04x `neon` |
| `hevc_encode_cabac` | 2.256 ms | 2.195 ms (1.03x) | 1.03x `neon` |
| `hevc_inter_pred` | 24.460 ms | 20.344 ms (1.20x) | 1.20x `neon` |
| `hevc_intra_pred` | 8.569 ms | 8.396 ms (1.02x) | 1.02x `neon` |
| `hevc_inverse_transform` | 8.278 ms | 7.636 ms (1.08x) | 1.08x `neon` |
| `hevc_sao` | 35.250 ms | 22.443 ms (1.57x) | 1.57x `neon` |

### Reading the sub-parity rows

An arm below `1.00x` is slower under its vector kernel than under scalar. Before
treating one as a defect, note what three independent measurement sets of this
same table did to the candidates:

| Group | set 1 | set 2 | set 3 |
| --- | ---: | ---: | ---: |
| `hevc_intra_pred` | 0.62x | 1.04x | 1.09x |
| `av1_intra_paeth` | 0.78x | 0.88x | 0.98x |
| `av1_forward_dct_32x32` | 0.78x | 0.91x | 0.95x |

No kernel changed between set 2 and set 3. What changed was the load average on
the measuring host, and every candidate walked towards parity as the machine got
quieter. **On this host, at this noise level, no arm is reliably below parity
except the ones with no vector kernel at all** — `av1_encode_stage_symbol`,
`av1_encode_stage_bitstream`, `hevc_cabac`, `hevc_encode_cabac`,
`av1_entropy_symbol` and `hevc_color_convert`, where the two arms are the same
code and differ only by measurement noise. Of those, `hevc_color_convert` is the
one worth acting on, and `#219` already tracks vectorizing it.

The rest of the near-parity rows are the story this file tells above: under
`lto = "fat"` with `codegen-units = 1`, LLVM does to the scalar reference roughly
what the hand kernel does, and the two land within noise of each other.

This is the single most useful thing the committed table records. A one-off
measurement of any of the three rows above would have looked like a broken
kernel and sent someone rewriting code that was fine. It is also why the CI job
compares medians rather than means, sets its threshold at a deliberately loose
15%, and reports instead of failing: a shared runner is a noisier host than this
one, not a quieter one.

An arm being absent from a row means the host could not execute it, not that it
was not measured: an Apple Silicon host has no `sse41` or `avx2` column at all,
which is why the x86_64 arms do not appear here yet. `#228` covers measuring the
x86_64 side.

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
| `..._reconstruct` | encode-side reconstruction (predict + add residual per coded block) plus the §8.7.2 deblocking filter and §8.7.3 SAO over the reconstructed picture |
| `..._pcm_write` | whole-picture access-unit writing: parameter sets, slice header, CABAC-coded CU syntax, PCM samples |
| `hevc_encode_cabac` | the §9.3.5 arithmetic encoder alone, over a synthetic bin stream of single, interleaved bins |
| `hevc_encode_cabac_bypass` | the same encoder over contiguous *bypass runs* — the shape 62% of the lossy residual writer's bins have |
| `hevc_encode_bitwriter` | the raw fixed-length / `ue(v)` / `se(v)` writer alone |
| `..._rgba_to_yuv420` | the RGBA8 input conversion every encoded frame pays (`engine::encoder::colorconv`) |

Every group runs once per available instruction set through
`support::isa::bench_across_isas`, with the same bit-exactness and
`active_by_site()` guards as the decode-side groups.

### Resolutions

640x352 always; 1920x1088 behind `ZVIDLIB_BENCH_LARGE=1`. The PCM writer
requires dimensions divisible by 16, so the nominal 640x360 and 1920x1080 are
not encodable — these are the nearest valid sizes at the same scale.

### Stages this encoder does not have yet

The encoder is a lossless PCM bootstrap writer. It has **no forward transform
and no quantization**: PCM samples are written verbatim, so there is no residual
to transform or quantize. Those stages have no group here because they do not
exist yet, and the target prints that on every run so a missing group is never
read as a stage that costs nothing.

`..._reconstruct` does exist, and it is the one stage whose measured shape
depends on the access unit being modelled. The reconstruction loop always runs
(predict, add the coded residual, clip); the in-loop filters only modify samples
when the access unit leaves them enabled on its PCM coding units
(`pcm_loop_filter_disabled_flag == 0`), which is the shape this group models,
because the filters are what it exists to measure. The shipped writer neutralizes
them, which is what keeps its PCM encode exactly lossless.

### Where the SIMD axis reads flat, and why that is the result

The encoder has three SIMD dispatch families of its own: `hevc_rdcost`, the SAD
and SATD distortion metrics the mode search calls; `hevc_fwd_transform_quant`,
the forward transform and quantization; and `hevc_colorconv`, the RGBA8 to
YUV420 input conversion. A fourth group, `..._reconstruct`, also moves with the
instruction set, but by reaching the decoder's already-vectorized deblocking and
SAO kernels rather than an encoder-side one.

**Bitstream writing and CABAC** have no vector path at all, so `..._pcm_write`,
`hevc_encode_cabac`, `hevc_encode_cabac_bypass` and `hevc_encode_bitwriter` are
expected to read the same under every instruction set. That is a measured result, not a broken benchmark:
it says the remaining encoder-side vectorization targets are entropy coding and
the bitwriter — and that for the bitwriter, a widening rewrite of its
`put_bit`-at-a-time inner loop is likely worth more than vector kernels, while
CABAC's renormalization is serial by construction and is a
bin-parallel-algorithm question rather than a kernel one. It is also why every
group asserts through `simd::active_by_site()` that the override landed rather
than inferring it from the clock — see
[Reading a null result](#reading-a-null-result).

That widening rewrite has since happened, and `..._pcm_write` and
`hevc_encode_bitwriter` still read flat on the SIMD axis — deliberately. The
rewrite widened `BitWriter::put_bits` to move a chunk of a field at a time and
gave byte-aligned §7.3.8.7 PCM sample data a bulk path that bypasses the bit
accumulator entirely; neither is a vector kernel, so neither added a
`simd::active_by_site()` site and neither arm moves when the instruction set is
pinned. **A flat SIMD arm here does not mean no work was done.** The win is on
the scalar axis and has to be read as a before/after against the previous
implementation rather than as a scalar-versus-NEON ratio within one run:
measured on a contended Apple Silicon host as the best of seven interleaved
rounds per arm, `hevc_encode_bitwriter` went 3.43 ms -> 0.58 ms (~5.9x) and
`hevc_encode_640x352_pcm_write` 8.32 ms -> 0.35 ms (~23x), with the scalar and
NEON arms of each group staying level with each other throughout, exactly as
this section predicts. Output is byte-identical either way; `tests/hevc_
bitstream_byte_identity.rs` pins the produced access units to digests captured
from the pre-rewrite writer.
### Why the CABAC arithmetic encoder stays serial

`hevc_encode_cabac` reads flat under every instruction set and is expected to
keep doing so. The §9.3.5 arithmetic encoder is serial by construction — each
bin's renormalization reads the interval the previous bin left — so nothing
vectorizes here and any speedup would have to come from a bin-parallel or
speculative *algorithm*. It was measured to decide whether such an algorithm is
worth its complexity. It is not, and this section is the record of why, so the
question does not have to be re-derived.

Two things were measured on an Apple Silicon host, both as the best of nine
interleaved rounds per arm and reproduced across three separate runs, because
this machine routinely runs several concurrent builds and a single run of
anything on it is worth several times its own value in noise.

**First, where a bin's time goes.** The same 262,144-bin mixed workload
`hevc_encode_cabac` uses — alternating context-coded and bypass bins over 64
context models — run through the shipped engine with its bit sink progressively
removed:

| Arm | Throughput | vs shipped |
| --- | --- | --- |
| the shipped engine, writing through `BitWriter` | 74.5 Mbin/s | 1.00x |
| the same engine over a word-accumulating bit sink | 84.9 Mbin/s | 1.14x |
| the same engine emitting no bits at all | 129.8 Mbin/s | 1.75x |
| the per-context §9.3.4.3.2.2 state transitions alone | 275.5 Mbin/s | 3.70x |

The last row is the ceiling on the whole question. A bin-parallel formulation
attacks the interval arithmetic and the renormalization; the table prices making
*all* of it, and every bit of output with it, entirely free at **3.7x** — and
what is left over at that point is another serial dependence chain, because a
context's state after bin *k* is what bin *k+1* codes against. Every one of
those 3.7x has to be paid for in speculation, rollback, and the merging of
per-segment carries and outstanding-bit runs, whose per-bin bookkeeping is
comparable to the four table lookups and two shifts it would be replacing.

**Second, what the stage is worth at the frame level.** Ablating the three
CABAC entry points out of the access-unit writers at 640x352, so every other
stage still runs:

| Writer | Per access unit | CABAC's share | Bins per access unit |
| --- | --- | --- | --- |
| `encode_idr_pcm_au` (lossless) | 10.5–11.7 ms | 4–10% | 55,068 |
| `encode_idr_residual_au` (lossy, QP 26) | 37.5–42.1 ms | 64–66% | 1,472,210 |

For scale, the mode search over the same picture is 1.6–1.7 ms intra and
39–42 ms inter per frame.

This corrects the premise the question was filed under. On the lossless PCM path
CABAC really is a rounding error — the bins are a few per coding unit and the
other 90%+ of that writer is raw sample bytes going through the
`put_bit`-at-a-time loop, which is the bitwriter's problem and not the arithmetic
coder's. But on the lossy residual path the encoder gained in #227, CABAC is
about **two thirds** of the access-unit write, so the stage is worth working on
after all. What does not follow is that bin-parallelism is the way to work on
it.

**What to do instead.** Two ordinary serial changes were expected to take most
of the reachable headroom with no speculation. The first was tried and does not;
the second has not been tried yet.

1. **Encoding a bypass *run* in one step is an identity, and is not a speedup**
   (#246). §9.3.5.5 unrolled over `n` bins is
   `ivlLow = ( ivlLow << n ) + ivlCurrRange * value` followed by the same
   carry-controlled emission of the top `n` bits, each tested against a window
   scaled by the bins still below it. That part holds exactly: implemented and
   run against the bin-at-a-time engine it is byte-identical, and leaves an
   identical `ivlLow` and `bitsOutstanding`, for every run length from 1 to 32
   from every primed engine state, and the lossy residual writer's output does
   not move by a bit at any QP. What did not reproduce is the **1.73x** this
   section previously recorded for it. Measured as the best of 25 rounds of the
   two paths interleaved *in one process* — the only form of this comparison
   that survives a host running several concurrent builds — over
   `hevc_encode_cabac_bypass`'s 262,144-bin workload:

   | Run length | bin at a time | run at a time | ratio |
   | --- | --- | --- | --- |
   | 2 | 182.8 Mbin/s | 166.7 Mbin/s | 0.91x |
   | 4 | 278.1 Mbin/s | 249.1 Mbin/s | 0.90x |
   | 8 | 281.7 Mbin/s | 298.0 Mbin/s | 1.06x |
   | 16 | 135.4 Mbin/s | 132.4 Mbin/s | 0.98x |
   | mixed 1–16 | 120.0 Mbin/s | 113.7 Mbin/s | 0.95x |

   The ratio straddles 1.00 and never leaves the noise, so the unrolled step
   was not taken. The reason it buys nothing is that it removes a shift and a
   conditional add per bin, and neither is what a bypass bin costs: the
   three-way, data-dependent Figure 9-13 decision — emit a bit, emit a
   carry-resolving bit, or defer one — still runs once per bin, and the
   `put_bit`-at-a-time sink under it still writes one bit at a time. Batching a
   whole run's resolved bits into a single `BitWriter::put_bits`, over a
   byte-at-a-time `put_bits`, was measured too and did not move the ratio
   either.
2. **Widen the bit sink**, which the first table prices at 1.14x for this stage
   on its own. That is the rewrite the `..._pcm_write` and
   `hevc_encode_bitwriter` groups took (#233, recorded above), it is what the
   run-at-a-time result above points back at, and the arithmetic encoder keeps
   only the share of it that its own bit writing is worth.

Neither is a bin-parallel algorithm, and neither changes a single bit of output.
Whatever headroom a speculative formulation might still hold after both is
smaller than the 3.7x ceiling above and costs far more to hold onto, which is
why the arithmetic encoder stays serial. `hevc_encode_cabac_bypass` exists so
that the run-at-a-time question stays measured rather than re-derived: it is the
same engine over contiguous bypass runs instead of single interleaved bins, and
it is the group any future attempt at this has to move.

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
