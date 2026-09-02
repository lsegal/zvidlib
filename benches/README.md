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
| `av1_encode_stage_iwht` | the lossless inverse 4x4 WHT, `src/av1_encoder/wht.rs` |
| `av1_encode_stage_symbol` | symbol coding over the static CDF tables, `src/av1_encoder/symbol.rs` and `cdf.rs` |
| `av1_encode_stage_coeff_ctx` | the §8.3.2 `coeff_base`/`coeff_br` context derivation on its own, `src/av1_simd/coeff.rs` |
| `av1_encode_stage_tile` | tile encoding: superblock iteration, `DC_PRED`, coefficient coding and its vectorized §8.3.2 context derivation, `src/av1_encoder/tile.rs` and `src/av1_simd/coeff.rs` |
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
coding, not the transform, is what a faster lossless encoder has to attack — and
the §8.3.2 context derivation half of it is now vectorized, which
`av1_encode_stage_coeff_ctx` measures directly. `av1_encode_stage_tile` still
reads close to flat anyway; [Why the tile group barely moves](#why-the-tile-group-barely-moves)
is the measurement that says why, and what the remaining target is.

This encoder's vectorized kernels are the forward transforms, the forward WHT
and its inverse — both WHT directions on `neon` only, since each was measured
under parity on x86_64 and routed to the scalar reference there — and the
`coeff_base` / `coeff_br` context derivation the coefficient coding loop runs on (§8.3.2, `src/av1_simd/coeff.rs`, the `av1_coeff_ctx` dispatch site).
The last of those derives a whole block's contexts in one data-parallel pass
ahead of the serial symbol loop, which is legal because the loop walks the
up-right diagonal scan backwards, so every neighbour a position consults is
already final — or zero, past the end-of-block. `av1_encode_stage_coeff_ctx` is
that pass on its own, and is the group its scalar-versus-vector delta is visible
in.

Symbol coding itself, CDF handling and bitstream writing remain scalar and are
expected to stay that way — the range coder is serial by construction, since
every symbol updates the CDF and the coder state the next symbol is written
against — so `av1_encode_stage_symbol` and `av1_encode_stage_bitstream` read the
same under every instruction set. That flatness is a measured result rather than
a broken run, which is why
each group asserts through `simd::active_by_site()` that the override landed
instead of inferring it from the clock. `report_stage_coverage` prints the stage
list on every run, so a group that stops being measured reads as a broken run
rather than as a stage that costs nothing.

```sh
cargo bench --bench av1_encode -- av1_encode_stage    # the per-stage groups only
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench av1_encode  # add the 1080p pass
```

### Why the tile group barely moves

`av1_encode_stage_coeff_ctx` wins and `av1_encode_stage_tile` does not, and the
gap between those two facts is the useful measurement here.

On an Apple Silicon host at 640x352, `--release`, as the best of interleaved
rounds per arm — this machine routinely runs several concurrent builds, so a
single round is meaningless and the minimum is the statistic (see
[Reading a null result](#reading-a-null-result)):

| Group | scalar | neon |
| --- | --- | --- |
| `av1_encode_stage_coeff_ctx` | 3.05 ms | 1.69 ms |
| `av1_encode_stage_tile` | 33.8 ms | 31.3 ms |

The kernel is 1.8x, and it is real. It is also only about 9% of the tile encode
it was factored out of, so removing 45% of *that* is under 4% end to end — below
this host's round-to-round spread, which is why `av1_encode_stage_tile` and
`av1_encode_frame_q0` still read within noise of each other and of the same two
groups built from `main`.

A `sample` profile of the lossless tile encode said where the rest went: of
14,411 samples inside `FrameEncoder::encode`, 9,164 — 64% — were in
`SymbolEncoder::encode_symbol`. Coefficient coding is indeed the whole frame,
but what is left of it once the contexts are vectorized is the range coder, and
that is serial by construction in exactly the way
[the HEVC CABAC encoder](#why-the-cabac-arithmetic-encoder-stays-serial) is:
each symbol renormalizes against the interval the previous one left. No vector
kernel addresses it. The remaining headroom in AV1 lossless encoding was
therefore a *serial* question — widening the bit sink, and unrolling the
literal-bit runs the coefficient loop writes — not another dispatch family.

#### What the serial work bought

Both of those changes have since been made, and the 64% figure above is the
state before them rather than the state now. `src/av1_encoder/symbol.rs` no
longer buffers each output byte as a `u16` and resolves the pending carries in a
second reverse pass over the whole stream at `finish`; it normalizes each byte
as it arrives, which halves the sink and drops the pass. And the equiprobable
literal bits — coefficient signs, the `eob` extra bits, the exp-Golomb tails —
are coded as runs rather than one `encode_symbol` call per bit, against a
specialization of the interval update for the one CDF `read_bool` ever uses. The
Golomb tail in particular is a single run: `len - 1` zeros followed by `x` in
`len` bits is just `x` written as one `2 * len - 1`-bit field.

Same host, same method — `--release`, best of interleaved rounds per arm, five
rounds at 640x352 and seven at 1920x1080, against the `main` these numbers were
measured next to:

| Group | main | with the serial work | |
| --- | --- | --- | --- |
| `av1_encode_stage_symbol` | 2.20 ms | 1.77 ms | 1.24x |
| `av1_encode_stage_tile/neon` | 36.5 ms | 33.0 ms | 1.11x |
| `av1_encode_frame_q0/neon` | 38.4 ms | 34.9 ms | 1.10x |
| `av1_encode_stage_symbol_1080p` | 21.4 ms | 16.7 ms | 1.29x |
| `av1_encode_stage_tile_1080p/neon` | 400 ms | 310 ms | 1.29x |
| `av1_encode_frame_q0_1080p/neon` | 442 ms | 318 ms | 1.39x |

The two `stage_symbol` rows are quoted as the best of both arms rather than one
of them, because that group is scalar on both and its arms differ only by this
host's spread; the tile and whole-frame rows are the `neon` arm, which is what
this host actually runs.

The shape of the result is the point. This is the first change to move
`av1_encode_stage_tile` and `av1_encode_frame_q0` at all — the vectorized
contexts could not, at 9% of a tile encode — and it moves them by roughly what
their range-coder share predicts, which is the confirmation that the profile was
reading the right thing. It also does not touch the bitstream: every group's
`bench_across_isas` guard sees the same output byte count it saw before
(264,392 for a 640x352 tile, 2,474,396 for a 1080p frame), and the encoder's
`a_fixed_frame_encodes_to_the_same_bytes_on_every_host` digests are unchanged.
The range coder is still serial, and still the largest single item in a lossless
frame; it is now a smaller one.

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
| `hevc_inter_pred` | §8.5.3.3 interpolation + the weighted combine, over the measured prediction-unit mix | yes |
| `hevc_intra_pred` | §8.4.4.2 reference smoothing, planar / DC / angular | yes |
| `hevc_deblock` | §8.7.2 luma block-edge deblocking | yes |
| `hevc_sao` | §8.7.3 sample adaptive offset, over the measured per-CTB parameter mix | yes |
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
single reason, and then, one stage at a time, why each of the other isolated
figures did not predict what that stage moved (#280 for inter prediction, #310
for SAO).

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
4:2:0) on an Apple Silicon host (M1, 8 cores), `--release`, minimum of six
interleaved rounds per arm. Both arms are reported side by side because since
issue #219 gave `color_convert` a kernel of its own they no longer agree to
within a point: the `scalar` arm runs 27.23 ms/frame and the `neon` arm 19.32
ms/frame, a 1.41x whole-frame ratio.

`scalar` arm, 27.23 ms/frame:

| Stage | Share of total | Share of decode | ms/frame | Vectorized |
| --- | ---: | ---: | ---: | --- |
| `color_convert` | 33.6% | n/a | 9.16 | yes |
| `inter_pred_filter` | 16.5% | 24.9% | 4.46 | yes |
| `sao_filter` | 2.4% | 3.6% | 0.72 | yes |
| `sao_snapshot` | 1.6% | 2.4% | 0.49 | no |
| `sao_setup` | 0.1% | 0.1% | 0.02 | no |
| `deblock` | 8.2% | 12.4% | 2.24 | yes |
| `intra_pred` | 4.0% | 6.0% | 1.08 | yes |
| `residual_cabac` | 3.8% | 5.8% | 1.04 | no |
| `inter_pred_write` | 3.6% | 5.5% | 0.98 | no |
| `motion_derive` | 3.3% | 5.0% | 0.90 | no |
| `inverse_transform` | 3.0% | 4.5% | 0.81 | yes |
| `slice_data_cabac` | 2.1% | 3.2% | 0.57 | no |
| `dpb_output` | 1.8% | 2.7% | 0.46 | no |
| `inter_pred_setup` | 0.9% | 1.4% | 0.25 | no |
| `header_parse` | 0.0% | 0.0% | 0.00 | no |
| _unattributed_ | 9.5% | 14.3% | 2.57 | n/a |

`neon` arm, 19.32 ms/frame:

| Stage | Share of total | Share of decode | ms/frame | Vectorized |
| --- | ---: | ---: | ---: | --- |
| `inter_pred_filter` | 21.5% | 23.8% | 4.15 | yes |
| `sao_snapshot` | 2.7% | 3.0% | 0.52 | no |
| `sao_filter` | 1.9% | 2.1% | 0.36 | yes |
| `sao_setup` | 0.1% | 0.1% | 0.02 | no |
| `deblock` | 11.0% | 12.1% | 2.13 | yes |
| `color_convert` | 9.6% | n/a | 1.86 | yes |
| `intra_pred` | 5.5% | 6.2% | 1.08 | yes |
| `residual_cabac` | 5.4% | 6.0% | 1.04 | no |
| `inter_pred_write` | 5.2% | 5.7% | 0.97 | no |
| `motion_derive` | 4.7% | 5.2% | 0.89 | no |
| `inverse_transform` | 3.8% | 4.2% | 0.73 | yes |
| `slice_data_cabac` | 2.9% | 3.2% | 0.57 | no |
| `dpb_output` | 2.5% | 2.7% | 0.46 | no |
| `inter_pred_setup` | 1.3% | 1.4% | 0.25 | no |
| `header_parse` | 0.0% | 0.0% | 0.00 | no |
| _unattributed_ | 13.2% | 14.6% | 2.56 | n/a |

"Share of decode" divides by the total minus `color_convert`, because colour
conversion is not decoding: it is the YUV420-to-RGBA pass every whole-frame
measurement takes on the way out of the decoder. Both denominators are reported
because the two answer different questions and are easy to confuse.

The two `sao` tables above are the arms measured *after* issue #310; the rest of
each table is the issue #280 measurement set, so the SAO rows are the only ones
in them taken on a different day. What they replaced was a single `sao` row at
2.57 ms/frame `scalar` against 2.59 `neon` — 14.4% and 14.7% of decode proper,
at 0.99x — which is the reading issue #310 asked about; see below.

§8.7.3 SAO is three rows rather than one as of issue #310, for the same reason
inter prediction is: only one of the three reaches a vector kernel. `sao_filter`
is the §8.7.3.2 per-CTB modification process, the band and edge classifiers the
`engine::simd::in_loop` kernels are; `sao_snapshot` is the §8.7.3.1
`saoPicture = recPicture` whole-picture copy the classification reads against;
`sao_setup` is the §7.4.9.3 `SaoOffsetVal` resolution over the CTB grid and the
§8.7.3.2 boundary grids built for it.

§8.5.3.3 inter prediction is three rows rather than one as of issue #280, which
is most of what that issue turned out to be about — see below. `inter_pred_filter`
is the interpolation and the weighted combine, the part `engine::simd` is;
`inter_pred_write` is the §8.4.4.1 `Clip1( pred + res )` write-back into the
picture; `inter_pred_setup` is the §8.5.3.3.2 reference-plane setup and the
per-prediction-unit allocation the other two happen between. Only the first
reaches a vector kernel.

_unattributed_ is real work in no instrumented scope — the coding-quadtree and
CTU walks, the per-CU residual extraction glue, and allocation between stages.
It is left as its own row rather than spread across the stages, so no share is
inflated by work it does not do. The profiler's own cost is under 3% of the
total at ~492k scopes; the example prints that bound on every run.

#### What it says

**The issue's hypothesis was wrong.** Entropy decoding is not where the time
goes. `slice_data_cabac` and `residual_cabac` together are **9.2% of decode
proper on the `neon` arm** — the §9.3.4 arithmetic decoder is serial and has no
vector path, but it is nowhere near large enough to be the reason whole-frame
SIMD reads flat.

**Vectorized stages cover 65.0% of the measured total and 61.2% of decode
proper.** By Amdahl, infinitely fast vector kernels would move the measured
whole-frame number 2.85x, a uniform 2x on those stages gives 1.48x, and a
uniform 4x gives 1.95x. So the kernels are *not* a minority of decode time.

**The largest item was not decoding at all, and it now has a kernel.**
`color_convert` — the per-sample BT.601/709 integer conversion in
`picture_to_rgba` — was a third of everything the whole-frame groups measure
with no vector path whatsoever, so every whole-frame SIMD number was diluted by
roughly a third for a stage no HEVC kernel touched. Issue #219 vectorized it
(`src/hevc/color_convert.rs`, timed by the `hevc_color_convert` group), and it
falls from **9.16 ms/frame to 1.86 ms/frame — 4.9x** — which is most of why the
whole-frame ratio moved from ~1.06x to 1.41x on this host. It is still a third
of the `scalar` arm, because that arm is what a *scalar* colour conversion
costs; on the `neon` arm it is 9.6%.

#### `inter_pred`: the isolated ratio and the in-decode one, reconciled

Issue #280 asked why §8.5.3.3 inter prediction — the largest stage of decode
proper on the `neon` arm — moves the whole-frame arms so much less than its
isolated kernel numbers suggest. The answer is two things, both measured on this
host, and neither of them a slow kernel.

**A third of the stage was never a kernel.** What this file used to report as one
`inter_pred` row, at 32.6% of decode proper and marked "vectorized", is the three
rows above. Before the repair below, on the `neon` arm: `inter_pred_filter` 4.18
ms/frame, `inter_pred_write` 1.89 ms/frame, `inter_pred_setup` 0.21 ms/frame —
34.2% of decode proper, of which **11.4 points, a third of the stage, reached no
vector kernel at all**. An isolated group that times only the kernels cannot
predict a stage ratio a third of which is fixed cost, however accurate it is
about the kernels: with the kernel at 1.25x and 33% of the stage invariant, the
stage's ceiling is 1.15x.

The write-back was also the cheapest thing here to fix. `Clip1( pred + res )` was
a per-sample loop calling `Picture::set_sample`, which re-resolved the plane and
re-derived its stride for every output sample, with the `Option` residual
branched on per sample. Resolving the plane once per prediction unit and hoisting
the residual branch out of the row loop leaves two row slices of known equal
length that LLVM vectorizes on its own: **1.89 → 0.97 ms/frame on the `neon` arm
(1.95x) and 1.84 → 0.98 on the `scalar` arm (1.88x)**, taking the whole decode
from 20.33 to 19.32 ms/frame on `neon` (5.0% faster) and 28.10 to 27.23 on
`scalar`, and the whole-frame ratio from 1.38x to 1.41x. The arithmetic is
unchanged and `tests/codec_conformance.rs` passes on its committed per-frame
SHA-256 digests, which is what says the samples written are the same ones.

**The isolated benchmark was measuring the wrong blocks.** `hevc_inter_pred` ran
a uniform grid of 16x16 bi-predicted *luma-only* blocks. A real decode does not:
48 frames of the bundled sample reconstruct 89,213 prediction units over
100,156,544 luma samples, and weighted by sample they are **62.1% 64x64, 31.1%
32x32, 5.3% 16x16 and 1.4% 8x8** — 61.6% bi-predicted, 38.4% uni-predicted, and
at 4:2:0 every luma sample brings half a chroma sample through the
§8.5.3.3.3.3 4-tap filter. (The clip codes 2Nx2N units throughout, so every size
is square; a stream using the §7.3.8.5 asymmetric partitions would add
rectangular units and this would be re-measured.)

That matters because **the 8-tap kernel's advantage over the auto-vectorized
scalar reference is a function of block size**, and it runs the wrong way from
what the ticket assumed. Timing the same workload restricted to one size at a
time, all bi-predicted, luma only, minimum of three interleaved rounds per arm:

| Luma block | `scalar` | `neon` | ratio | share of real luma samples |
| --- | ---: | ---: | ---: | ---: |
| 8x8 | 515.10 µs | 359.47 µs | 1.43x | 1.4% |
| 16x16 | 1.2935 ms | 1.0343 ms | 1.25x | 5.3% |
| 32x32 | 5.8375 ms | 5.2920 ms | 1.10x | 31.1% |
| 64x64 | 10.121 ms | 9.7011 ms | 1.04x | 62.1% |

This is the same effect the `engine::simd` table already records from the other
direction: `filter_taps` reads 1.6-1.9x on aarch64 in the *block* path and ~1.0x
over one long L1-resident buffer. The old grid was the second-best size on that
table and carried 5.3% of the real work. #280 read the walk between those two
figures as the `w x ( h + 7 )` intermediate the two-dimensional path
materializes turning a large block into the buffer case; issue #309 measured it
and that is not what it is. See below.

#### What the block-size decay actually is (issue #309)

#280 inferred the intermediate from the shape of the sweep without instrumenting
the two passes separately. Doing so refutes it, three ways, all on the same
aarch64 host in one process per measurement:

**The decay is there with no intermediate.** `measure_interp_pass_split` times
the two passes apart. The horizontal pass alone — which writes a fresh buffer
nothing has yet read — decays 2.12x at 8x8, 1.53x at 16x16, 1.06x at 32x32,
1.03x at 64x64, and the one-dimensional `x_frac == 0` / `y_frac == 0` phases,
which build no intermediate at all, decay with it (horizontal-only 1.91x → 1.29x,
vertical-only 1.38x → 1.21x). Whatever erodes the ratio is fully present when
there is no intermediate to blame.

**The variable is the per-call row length.** `measure_filter_taps_by_row_length`
strips out the block walk, the allocations and the intermediate entirely: every
row length reads the same 4 KiB L1-resident tap buffer, writes the same output
buffer, and covers the same total sample count, so call size is the only thing
that moves. It reproduces the whole decay:

| Samples per `filter_taps` call | `scalar` | `neon` | ratio |
| --- | ---: | ---: | ---: |
| 4 | 12.99 ms | 3.38 ms | 3.84x |
| 8 | 7.50 ms | 2.33 ms | 3.21x |
| 16 | 3.96 ms | 1.92 ms | 2.06x |
| 32 | 2.46 ms | 1.64 ms | 1.50x |
| 64 | 1.92 ms | 1.44 ms | 1.33x |
| 128 | 1.62 ms | 1.37 ms | 1.19x |
| 256 | 1.46 ms | 1.33 ms | 1.10x |

The block walk issues one `filter_taps` call per output row, so the call size
*is* the block width, and the sweep over block sizes is this sweep. At 64x64 the
intermediate is 64 x 71 x 4 = 18 KiB, which fits inside the M1's 128 KiB L1D; it
was never spilling to begin with.

**The ratio falls because the scalar reference gets better, not because the
kernel gets worse.** Both arms improve with row length, but the auto-vectorized
scalar loop improves 8.9x across the sweep (12.99 → 1.46 ms) against the kernel's
2.5x (3.38 → 1.33 ms), and they converge. At 256-sample rows the kernel runs
4.19M outputs x 8 taps — 8.4M `vmlaq_s32` — in 1.33 ms, about 6.3 G/s, or two per
cycle at 3.2 GHz. That is the M1's integer SIMD multiply throughput. **The kernel
is at the hardware limit at large block sizes, and the scalar loop reaches the
same limit once its trip count is long enough to amortize its prologue.** The
1.4-3.8x at small sizes is the scalar call's per-invocation setup, not kernel
headroom that large blocks lose.

##### The two repairs that were tried, and lost

Tiling the two-dimensional path is what #309 proposed. It was implemented as a
wrap-around ring carrying only the `N` horizontal rows the vertical pass has
live, instead of all `h + N − 1` — the same total horizontal work with no
redundant re-filtering, and a 2 KiB working set at 64 wide. A/B'd against the
full-height intermediate in one process (`measure_2d_ring_vs_flat`, best of 15
interleaved rounds), it lost at every size, on both arms:

| Luma block | flat `neon` | ring `neon` | speedup | flat `scalar` | ring `scalar` | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8x8 | 8.87 ms | 15.79 ms | 0.56x | 17.89 ms | 19.96 ms | 0.90x |
| 16x16 | 4.91 ms | 6.07 ms | 0.81x | 8.62 ms | 9.44 ms | 0.91x |
| 32x32 | 3.40 ms | 3.69 ms | 0.92x | 4.69 ms | 4.87 ms | 0.96x |
| 64x64 | 2.89 ms | 2.94 ms | 0.98x | 3.33 ms | 3.36 ms | 0.99x |

Both arms are spelled out inside the test rather than one of them being the
production path, and the test asserts they agree sample-for-sample with each
other and with `interp_block` before it times anything, so the comparison stays
runnable and honest after the revert.

The modular slot index costs more than the intermediate does, and it replaces the
flat buffer's constant row stride — which LLVM strength-reduces — with an address
that jumps. Reverted; the flat intermediate stands.

Folding the kernel's mirror taps was tried next, since the half-pel luma phase
`[-1, 4, -11, 40, 40, -11, 4, -1]` and the half-pel chroma phase `[-4, 36, 36, -4]`
are palindromes, so each pair can share one multiply: 4 multiplies and 4 adds
instead of 8 multiplies, which should pay on an ALU-bound kernel. It measured
*slower* at every row length (1.51 against 1.33 ms at 256 samples, 4.99 against
3.38 at 4). Reverted.

##### What is left, and why it is not this issue

The remaining headroom is not in the kernel's instruction selection but in its
operand width. The reference plane, the tap slices and the intermediate are all
`i32`, four bytes per sample for eight-bit content, so every `vmlaq_s32` covers
four samples where an `i16` formulation would cover eight. That is a change to
how sample planes are represented across `engine`, not a change to `filter_taps`,
and it is tracked separately rather than folded in here.

Neither the decoded output nor any kernel changed in the course of this, so the
sweep, the `hevc_inter_pred` ratio and the `inter_pred_filter` ms/frame rows
above stand as re-measured.

Sample-weighting the sweep predicts 1.07x for the measured mix, and the rebuilt
group measures **1.09x against the old grid's 1.25x** on this host on the same
day (24.449 / 22.433 ms against 25.713 / 20.494 ms, minimum of four interleaved
rounds per arm). The other two differences turn out not to matter: at the
measured sizes, all bi-predicted and luma only reads 1.09x, adding the uni/bi
split reads 1.07x, and adding chroma reads 1.09x. **Block size accounts for the
whole of it.**

So the two measurements were consistent and the expectation was wrong. The
in-decode kernel arm reads 1.07x (4.46 / 4.15 ms/frame) and the isolated group,
now that it runs the blocks a decode actually runs, reads 1.09x. There is no
remaining gap between them to explain: what there was, was a benchmark timing
16x16 blocks for a decoder that spends 93% of its interpolation on 32x32 and
64x64 ones, plus a third of the stage that was never vectorized in the first
place.

#### `sao`: the isolated ratio and the in-decode one, reconciled

Issue #310 asked the same question of §8.7.3 SAO that #280 asked of inter
prediction: `hevc_sao` measured 1.57x in isolation while the `sao` row of the
breakdown read 2.57 ms/frame `scalar` against 2.59 `neon` — **0.99x** — on 14.7%
of decode proper. Two answers, and this time the first one is not a matter of
degree.

**No in-decode SAO sample reached a vector kernel at all.** §8.7.3.2's
per-sample edge classification has to deny a neighbour read that crosses a slice
or tile boundary with filtering across it disabled, so `apply_sao_ctb_full` took
its branch-free row path — the one that calls `sao_edge_row` / `sao_band_row` —
only when the caller passed no `SaoBoundaries` at all. The decoder always passes
one. It cannot do otherwise: whether a stream is single-slice and single-tile is
not known before it is parsed, and the same held for the `NoFilterMap` that
carries §8.7.3.1's PCM and transquant-bypass suppression, which is present for a
whole picture as soon as one coding unit anywhere in it qualifies. So every
picture the decoder filtered went down the per-sample scalar path, with a
`neighbour_allowed` CTB-grid lookup, an in-picture test and a `Picture::sample` /
`set_sample` plane resolution per sample — and both arms ran exactly the same
code. 0.99x was not a slow kernel or a diluted stage; it was an unreached one,
and no isolated measurement of the kernels could have predicted it.

Both tests are now asked about the CTB rather than about the picture.
`SaoBoundaries::ctb_neighbourhood_unconstrained` clears a CTB whose eight
neighbours are all mutually filterable — the classifier reads at most one sample
away, so no sample in such a CTB has a read the per-sample test could deny —
and `NoFilterMap::any_in_luma_rect` clears one no suppressed cell reaches. Band
offset (equation 8-414) classifies each sample by its own value and reads no
neighbour, so no boundary constraint applies to it at all. A CTB that fails
either test still takes the scalar path, which stays the normative reference; on
a single-slice single-tile picture none do.

**The isolated benchmark was filtering a picture a decode never filters.** The
old `hevc_sao` grid put band or edge offset on **every CTB of every component**.
Over the same 48 frames of the bundled sample, all 26,520 CTBs the decoder
resolved parameters for:

| | off | band | EO 0-deg | EO 90-deg | EO 135-deg | EO 45-deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| luma | **86.7%** | 2.0% | 2.1% | 3.1% | 2.7% | 3.5% |
| chroma | **94.4%** | 1.0% | 0.1% | 0.4% | 1.6% | 2.5% |

Cb and Cr came out identical CTB for CTB, as §7.3.8.3 signals one
`sao_type_idx_chroma` and one `SaoEoClass` for the pair. A `SaoTypeIdx == 0` CTB
is returned from before it reads a sample, so the old grid ran roughly nine times
the classifier work a 1080p frame does — and it ran it through a dispatch the
decoder never took. `SAO_CTB_MIX` is that table; the rebuilt group schedules to
it greedily, exactly as `INTER_PU_MIX` does, passes the single-slice single-tile
`SaoBoundaries` a decode carries, and folds only the switched-on CTBs, since the
rest of the picture is not something the stage produced.

**What the repair is worth.** 48 frames of the bundled 1920x1080 sample, Apple
Silicon (M1), `--release`, minimum of six interleaved rounds per arm, the two
builds run alternately in the same session:

| | `scalar` | `neon` | ratio |
| --- | ---: | ---: | ---: |
| `sao` before | 3.07 ms/frame | 3.10 ms/frame | 0.99x |
| `sao` after | 1.23 ms/frame | 0.90 ms/frame | 1.37x |
| whole frame before | 32.54 ms/frame | 23.46 ms/frame | 1.39x |
| whole frame after | 30.45 ms/frame | 21.95 ms/frame | 1.39x |

The stage falls **3.4x on the `neon` arm and 2.5x on the `scalar` arm**, taking
6.4% off a whole frame on both — the scalar arm gains too, because the row
kernels' scalar fallbacks are still branch-free rows against a per-sample loop
that re-resolved the plane for every sample. The whole-frame ratio is unchanged
at 1.39x, which is what a stage moving on both arms looks like. The host was
loaded while these were taken, so the absolute figures run above the `inter_pred`
set in the tables further up; the before/after comparison is valid because the
two builds were interleaved within it, and the arms are reported side by side
rather than against the committed table.

**And the two measurements now agree.** On the same host and the same day, in
interleaved rounds: the rebuilt `hevc_sao` group reads **1.24x** (2.659 /
2.149 ms) and the whole in-decode stage reads **1.37x**. The old grid re-measured
alongside them reads 1.18x rather than the 1.57x the committed table records,
which is the host, not a change — nothing in this work touches that build.

What is left between the two is one whole-picture copy. `apply_sao_picture_full`
snapshots `recPicture` so the classification always reads pre-SAO samples, and
the group clones a fresh input picture on top of that; the decode's stage carries
one such copy and the group carries two. Decomposed directly at 1920x1080 on the
`neon` arm: 0.56 ms for the group's input clone, 0.56 ms for the snapshot inside
the stage, ~0.75 ms for the classifiers, ~0.32 ms for the fold.

That decomposition is also the interesting thing the split now shows.
**`sao_snapshot` is larger than `sao_filter` on the `neon` arm** — 0.52 against
0.36 ms/frame — so the majority of what is left of SAO is a whole-picture `memcpy`
that reaches no kernel and never will. It is skipped outright when the resolved
grid switches SAO off everywhere, which on this content some pictures do, but the
general repair is a rolling band of pre-SAO rows rather than a picture, and that
is its own issue rather than part of this one.

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

Two hosts are recorded, not one, and they are deliberately not merged into a
single table. `bench_across_isas` runs one arm per entry in
`zvidlib::simd::available()`, so the arms a table has are a property of the
host that measured it: an absent column means "this CPU cannot execute that
instruction set", not "this row was skipped". An aarch64 baseline and an x86_64
one therefore describe disjoint halves of the dispatch matrix, and a ratio from
one says nothing about the other.

### Checking a table still describes the crate

A committed table is a measurement of a commit, and the commit is stamped on
it. What nothing checked until now is whether that commit's *kernels* are the
ones the crate has: a row silently stops describing anything the moment a
dispatch site lands under it, and the only signal was somebody noticing a ratio
that moved for no attributable reason. Chasing one such row down to a stale
table rather than a regression is the whole of what issue #361 turned out to
be, and `src/simd.rs` already refuses to let the *code* side of that go quiet —
`the_documented_site_table_lists_every_dispatch_site` makes a dispatch site
impossible to add without a check.

```sh
python3 .github/scripts/criterion_baseline.py staleness --readme benches/README.md
```

It reads each stamp out of this file, reads the dispatch sites
`zvidlib::simd::active_by_site` documents at that commit, diffs them against
the sites registered now, and names the rows whose subject site did not exist
when the table was drawn. Today that is three rows of the Apple M1 table —
`hevc_color_convert`, `av1_encode_stage_tile` and
`hevc_encode_640x352_reconstruct` — and nothing on the x86_64 one, which was
drawn at a commit with the same eleven sites the crate has now.

Three things about how it reads the site set are worth stating, because each is
a place a more obvious implementation does not work:

- The sites are read from `active_by_site`'s **rustdoc table**, not from a
  build. A dispatch site is a Rust value, and resolving it at a stamped commit
  would mean building that commit; the doc table is the same set by test, and
  it can be read out of a blob.
- A stamp is routinely a checkpoint commit on a branch whose ref is deleted at
  merge — both tables here are — so `git show` fails on exactly the commits
  this check exists to read. It falls back to the GitHub contents API, and a
  stamp neither can resolve is reported as *unverified* rather than as clean.
- A row is attributed to a site only when the row's group is that site's own
  number. Whole-frame groups like `hevc_encode_640x352` cross every site at
  once, so a landed kernel moves them by an amount no single row can be blamed
  for, and naming them would bury the rows that can be.

The check runs in the `Rust checks` job and writes its report to the step
summary. It reports and never gates, for the same reason [the delta
report](#the-threshold-and-why-it-is-only-a-report) does not: a stale table is
a measurement to redraw, not a broken build. The redraw is the recipe above.

It answers the site-set half of what can go stale. The `Vectorized` column in
[The HEVC per-stage groups](#the-hevc-per-stage-groups) is the other half, and
it is not checked here: a site can exist and still resolve to the scalar
reference on every arm — `hevc_recon` does — so "is there a dispatch site" and
"is there a vector kernel" are different questions, and only the first can be
answered from a commit that is not built.

### Apple M1 (aarch64)

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
| `hevc_inter_pred` | 24.449 ms | 22.433 ms (1.09x) | 1.09x `neon` |
| `hevc_intra_pred` | 8.569 ms | 8.396 ms (1.02x) | 1.02x `neon` |
| `hevc_inverse_transform` | 8.278 ms | 7.636 ms (1.08x) | 1.08x `neon` |
| `hevc_sao` | 2.659 ms | 2.149 ms (1.24x) | 1.24x `neon` |

#### Reading the sub-parity rows

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
`av1_encode_stage_bitstream`, `hevc_cabac`, `hevc_encode_cabac` and
`av1_entropy_symbol`, where the two arms are the same code and differ only by
measurement noise.

`hevc_color_convert` used to be listed there too, and this table's `1.27x` row
is what a same-code group reads at this noise level rather than a NEON win.
That is no longer what the group measures. `b6655bad215f` predates `f695a1a`,
the #222 merge that closed #219 by adding `src/hevc/color_convert.rs` with
scalar, SSE4.1, AVX2 and NEON backends, so this draw timed the old per-sample
scalar loop in `picture_to_rgba` on both arms. **The `hevc_color_convert` row
above is stale, and re-drawing this table is the only thing that will fix it**;
every other row in it is a group whose kernels are unchanged since the draw.

The rest of the near-parity rows are the story this file tells above: under
`lto = "fat"` with `codegen-units = 1`, LLVM does to the scalar reference roughly
what the hand kernel does, and the two land within noise of each other.

This is the single most useful thing the committed table records. A one-off
measurement of any of the three rows above would have looked like a broken
kernel and sent someone rewriting code that was fine. It is also why the CI job
compares medians rather than means, sets its threshold at a deliberately loose
15%, and reports instead of failing: a shared runner is a noisier host than this
one, not a quieter one.

### x86_64 with SSE4.1 and AVX2 (Linux)

Measured on a GitHub `ubuntu-latest` runner rather than on this project's
development machine, because no aarch64 host can produce these columns at all.
The rounds ran with `ZVIDLIB_BENCH_LARGE=1` and the elementwise minimum was
taken across three of them, exactly as the recipe above describes. That is also
why this table carries the `_1080p` rows the Apple M1 one does not.

GitHub's `ubuntu-latest` pool is not uniform, so the CPU model is checked before
a round is used: an elementwise minimum taken across different CPU models is
attributable to no named host, and the whole point of naming one is that the
numbers are not interchangeable. Six rounds were dispatched at once so that
three sharing a model could be selected afterwards; the rounds that landed on an
Intel Xeon 6973P-C, an Intel Xeon Platinum 8573C and an AMD EPYC 9V74 80-Core
were measured and discarded. Every merged round logged `scalar`, `sse4.1` and
`avx2` in its `# host instruction sets:` line, and `# dispatch site av1_simd:
avx2` in its per-site log.

This table replaces the one #261 recorded at `e115506f8bf6` on an AMD EPYC 9V74.
That draw predates the codegen repair in #337 (issue #336), so every `av1_*` row
in it timed `av1_simd` kernels whose `#[target_feature]` wrappers had degenerated
into tail calls to baseline-instruction-set copies — each intrinsic an
out-of-line `core_arch` call with its operand spilled to the stack. It recorded
`av1_deblock_wide` at 0.13x under `avx2` and `av1_forward_flipadst_16x16` at
0.20x. Those figures described the compiler's output, not the kernels, and the
kernels they described no longer exist.

That draw is older than its date suggests in one further way. `e115506f8bf6` is
a checkpoint commit on the #257 branch, and its merge base with `main` is
`b9995b1` (#254) — so it does not contain `f695a1a`, the #222 merge, even though
#222 landed on `main` fifty minutes before the checkpoint was written. That is
the whole of why `hevc_color_convert` moved; see [Reading the
rows](#reading-the-rows) below.

Measured on **AMD EPYC 7763 64-Core Processor (Linux, x86_64)**, at
`b284c38a6391` — the #337 merge `b233f0a74f88` plus the temporary six-round
workflow that measured it, which touches no crate code.

The two `av1_encode_stage_coeff_ctx` rows are the pre-#362 code and are kept as
the record of the defect that issue reports; the repair is measured under [The
#362 re-measurement](#the-362-re-measurement) below, and the kernel that
replaces the routing under [The #371 re-measurement](#the-371-re-measurement),
each on its own host and with its own provenance. Both rows' `avx2` column is
therefore two repairs out of date, in the direction of being too slow.

The two `hevc_encode_*_rdo_inter` rows are pre-#370 in the same way; their
repair is measured under [The #370 re-measurement](#the-370-re-measurement),
which landed on *this* host, so its `avx2` column is directly comparable with
the one here.

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `av1_cdef` | 89.226 ms | 38.801 ms (2.30x) | 31.241 ms (2.86x) | 2.86x `avx2` |
| `av1_deblock` | 21.506 ms | 3.974 ms (5.41x) | 3.424 ms (6.28x) | 6.28x `avx2` |
| `av1_deblock_boundary` | 370.488 µs | 75.674 µs (4.90x) | 78.676 µs (4.71x) | 4.90x `sse4.1` |
| `av1_deblock_chroma` | 16.036 ms | 6.091 ms (2.63x) | 6.336 ms (2.53x) | 2.63x `sse4.1` |
| `av1_deblock_wide` | 104.338 ms | 36.634 ms (2.85x) | 31.647 ms (3.30x) | 3.30x `avx2` |
| `av1_decode_frame` | 98.518 ms | 98.855 ms (1.00x) | 99.331 ms (0.99x) | 1.00x `sse4.1` |
| `av1_encode_frame_q0` | 22.029 ms | 18.612 ms (1.18x) | 18.816 ms (1.17x) | 1.18x `sse4.1` |
| `av1_encode_frame_q0_1080p` | 206.988 ms | 176.980 ms (1.17x) | 178.784 ms (1.16x) | 1.17x `sse4.1` |
| `av1_encode_frame_q160` | 284.968 ms | 193.387 ms (1.47x) | 185.702 ms (1.53x) | 1.53x `avx2` |
| `av1_encode_frame_q160_1080p` | 2.617 s | 1.780 s (1.47x) | 1.715 s (1.53x) | 1.53x `avx2` |
| `av1_encode_frame_q32` | 316.545 ms | 217.592 ms (1.45x) | 212.056 ms (1.49x) | 1.49x `avx2` |
| `av1_encode_frame_q32_1080p` | 2.902 s | 2.008 s (1.44x) | 1.942 s (1.49x) | 1.49x `avx2` |
| `av1_encode_stage_bitstream` | 14.023 µs | 13.480 µs (1.04x) | 14.136 µs (0.99x) | 1.04x `sse4.1` |
| `av1_encode_stage_bitstream_1080p` | 126.152 µs | 126.368 µs (1.00x) | 126.356 µs (1.00x) | 1.00x `avx2` |
| `av1_encode_stage_coeff_ctx` | 4.305 ms | 1.415 ms (3.04x) | 1.725 ms (2.50x) | 3.04x `sse4.1` |
| `av1_encode_stage_coeff_ctx_1080p` | 39.301 ms | 13.042 ms (3.01x) | 15.838 ms (2.48x) | 3.01x `sse4.1` |
| `av1_encode_stage_symbol` | 874.188 µs | 878.248 µs (1.00x) | 880.264 µs (0.99x) | 1.00x `sse4.1` |
| `av1_encode_stage_symbol_1080p` | 8.257 ms | 8.167 ms (1.01x) | 8.236 ms (1.00x) | 1.01x `sse4.1` |
| `av1_encode_stage_tile` | 21.340 ms | 18.051 ms (1.18x) | 18.248 ms (1.17x) | 1.18x `sse4.1` |
| `av1_encode_stage_tile_1080p` | 197.811 ms | 167.742 ms (1.18x) | 169.684 ms (1.17x) | 1.18x `sse4.1` |
| `av1_encode_stage_iwht` † | 331.600 µs | 399.630 µs (0.83x) | 371.650 µs (0.89x) | 0.89x `avx2` |
| `av1_encode_stage_iwht_1080p` † | 3.064 ms | 3.682 ms (0.83x) | 3.424 ms (0.89x) | 0.89x `avx2` |
| `av1_encode_stage_wht` | 436.654 µs | 371.908 µs (1.17x) | 372.555 µs (1.17x) | 1.17x `sse4.1` |
| `av1_encode_stage_wht_1080p` | 4.026 ms | 3.423 ms (1.18x) | 3.419 ms (1.18x) | 1.18x `avx2` |
| `av1_entropy_symbol` | 3.767 ms | 3.767 ms (1.00x) | 3.766 ms (1.00x) | 1.00x `avx2` |
| `av1_forward_adst_8x8` | 32.313 ms | 9.663 ms (3.34x) | 9.213 ms (3.51x) | 3.51x `avx2` |
| `av1_forward_dct_16x16` | 41.498 ms | 13.372 ms (3.10x) | 12.227 ms (3.39x) | 3.39x `avx2` |
| `av1_forward_dct_32x32` | 56.890 ms | 43.147 ms (1.32x) | 38.382 ms (1.48x) | 1.48x `avx2` |
| `av1_forward_dct_4x4` | 42.591 ms | 7.670 ms (5.55x) | 7.774 ms (5.48x) | 5.55x `sse4.1` |
| `av1_forward_dct_8x8` | 32.806 ms | 10.046 ms (3.27x) | 9.275 ms (3.54x) | 3.54x `avx2` |
| `av1_forward_flipadst_16x16` | 38.322 ms | 13.948 ms (2.75x) | 12.354 ms (3.10x) | 3.10x `avx2` |
| `av1_intra_directional` | 35.603 ms | 35.580 ms (1.00x) | 35.582 ms (1.00x) | 1.00x `sse4.1` |
| `av1_intra_paeth` | 3.106 ms | 3.163 ms (0.98x) | 2.953 ms (1.05x) | 1.05x `avx2` |
| `av1_intra_smooth` | 3.078 ms | 3.079 ms (1.00x) | 3.078 ms (1.00x) | 1.00x `avx2` |
| `av1_inverse_adst_8x8` | 34.830 ms | 22.214 ms (1.57x) | 21.862 ms (1.59x) | 1.59x `avx2` |
| `av1_inverse_dct_16x16` | 25.175 ms | 15.434 ms (1.63x) | 14.931 ms (1.69x) | 1.69x `avx2` |
| `av1_inverse_dct_32x32` | 21.299 ms | 13.771 ms (1.55x) | 13.410 ms (1.59x) | 1.59x `avx2` |
| `av1_inverse_dct_4x4` | 53.750 ms | 25.005 ms (2.15x) | 25.732 ms (2.09x) | 2.15x `sse4.1` |
| `av1_inverse_dct_64x64` | 27.276 ms | 18.297 ms (1.49x) | 17.765 ms (1.54x) | 1.54x `avx2` |
| `av1_inverse_dct_8x8` | 33.400 ms | 18.343 ms (1.82x) | 17.920 ms (1.86x) | 1.86x `avx2` |
| `av1_inverse_flipadst_16x16` | 28.593 ms | 19.693 ms (1.45x) | 18.760 ms (1.52x) | 1.52x `avx2` |
| `av1_mc_blend_mask` | 26.056 ms | 14.137 ms (1.84x) | 11.179 ms (2.33x) | 2.33x `avx2` |
| `av1_mc_compound_average` | 26.135 ms | 15.448 ms (1.69x) | 11.533 ms (2.27x) | 2.27x `avx2` |
| `av1_mc_single` | 13.372 ms | 6.742 ms (1.98x) | 5.044 ms (2.65x) | 2.65x `avx2` |
| `av1_motion_compensation` | 13.154 ms | 6.845 ms (1.92x) | 5.324 ms (2.47x) | 2.47x `avx2` |
| `av1_self_guided` | 10.743 ms | 3.859 ms (2.78x) | 3.077 ms (3.49x) | 3.49x `avx2` |
| `av1_wiener` | 11.655 ms | 8.438 ms (1.38x) | 6.213 ms (1.88x) | 1.88x `avx2` |
| `hevc_cabac` | 2.201 ms | 2.201 ms (1.00x) | 2.201 ms (1.00x) | 1.00x `avx2` |
| `hevc_color_convert` | 11.827 ms | 3.124 ms (3.79x) | 2.483 ms (4.76x) | 4.76x `avx2` |
| `hevc_deblock` | 14.063 ms | 13.462 ms (1.04x) | 13.451 ms (1.05x) | 1.05x `avx2` |
| `hevc_decode` | 700.695 ms | 599.583 ms (1.17x) | 578.868 ms (1.21x) | 1.21x `avx2` |
| `hevc_decode_to_picture` | 626.111 ms | 586.569 ms (1.07x) | 562.726 ms (1.11x) | 1.11x `avx2` |
| `hevc_encode_1920x1088` | 978.101 ms | 621.336 ms (1.57x) | 646.477 ms (1.51x) | 1.57x `sse4.1` |
| `hevc_encode_1920x1088_fwd_transform_quant` | 148.217 ms | 96.906 ms (1.53x) | 93.307 ms (1.59x) | 1.59x `avx2` |
| `hevc_encode_1920x1088_pcm_write` | 6.511 ms | 6.510 ms (1.00x) | 6.518 ms (1.00x) | 1.00x `sse4.1` |
| `hevc_encode_1920x1088_rdo_inter` | 851.069 ms | 520.861 ms (1.63x) | 547.481 ms (1.55x) | 1.63x `sse4.1` |
| `hevc_encode_1920x1088_rdo_intra` | 51.430 ms | 34.466 ms (1.49x) | 34.307 ms (1.50x) | 1.50x `avx2` |
| `hevc_encode_1920x1088_reconstruct` | 97.670 ms | 50.193 ms (1.95x) | 45.805 ms (2.13x) | 2.13x `avx2` |
| `hevc_encode_1920x1088_reconstruct_quantized` | 223.384 ms | 119.493 ms (1.87x) | 111.925 ms (2.00x) | 2.00x `avx2` |
| `hevc_encode_1920x1088_residual_write` | 2.332 s | 1.805 s (1.29x) | 1.687 s (1.38x) | 1.38x `avx2` |
| `hevc_encode_1920x1088_rgba_to_yuv420` | 5.822 ms | 1.172 ms (4.97x) | 928.249 µs (6.27x) | 6.27x `avx2` |
| `hevc_encode_640x352` | 101.668 ms | 64.354 ms (1.58x) | 67.101 ms (1.52x) | 1.58x `sse4.1` |
| `hevc_encode_640x352_fwd_transform_quant` | 15.923 ms | 10.409 ms (1.53x) | 9.841 ms (1.62x) | 1.62x `avx2` |
| `hevc_encode_640x352_pcm_write` | 726.265 µs | 727.603 µs (1.00x) | 726.733 µs (1.00x) | 1.00x `avx2` |
| `hevc_encode_640x352_rdo_inter` | 89.626 ms | 54.769 ms (1.64x) | 57.659 ms (1.55x) | 1.64x `sse4.1` |
| `hevc_encode_640x352_rdo_intra` | 5.518 ms | 3.707 ms (1.49x) | 3.700 ms (1.49x) | 1.49x `avx2` |
| `hevc_encode_640x352_reconstruct` | 10.120 ms | 5.109 ms (1.98x) | 4.748 ms (2.13x) | 2.13x `avx2` |
| `hevc_encode_640x352_reconstruct_quantized` | 23.300 ms | 12.547 ms (1.86x) | 11.732 ms (1.99x) | 1.99x `avx2` |
| `hevc_encode_640x352_residual_write` | 248.489 ms | 193.174 ms (1.29x) | 181.059 ms (1.37x) | 1.37x `avx2` |
| `hevc_encode_640x352_rgba_to_yuv420` | 645.519 µs | 135.292 µs (4.77x) | 106.489 µs (6.06x) | 6.06x `avx2` |
| `hevc_encode_bitwriter` | 703.320 µs | 703.788 µs (1.00x) | 703.451 µs (1.00x) | 1.00x `avx2` |
| `hevc_encode_cabac` | 1.683 ms | 1.694 ms (0.99x) | 1.686 ms (1.00x) | 1.00x `avx2` |
| `hevc_encode_cabac_bypass` | 2.053 ms | 2.052 ms (1.00x) | 2.054 ms (1.00x) | 1.00x `sse4.1` |
| `hevc_inter_pred` | 29.145 ms | 21.996 ms (1.32x) | 19.735 ms (1.48x) | 1.48x `avx2` |
| `hevc_intra_pred` | 8.304 ms | 8.296 ms (1.00x) | 7.887 ms (1.05x) | 1.05x `avx2` |
| `hevc_inverse_transform` | 9.291 ms | 7.079 ms (1.31x) | 6.396 ms (1.45x) | 1.45x `avx2` |
| `hevc_sao` | 32.116 ms | 19.431 ms (1.65x) | 18.325 ms (1.75x) | 1.75x `avx2` |

† The two `av1_encode_stage_iwht` rows come from a separate draw. The group did
not exist when the rest of this table was measured — #342 added it — so it was
measured on its own, by the same recipe: three rounds, elementwise minimum, on
one AMD EPYC 7763 64-Core, the model this table names. Six draws were dispatched
so that three sharing a model could be selected; two landed on an AMD EPYC 9V74
80-Core and were discarded. That draw timed `av1_encode_stage_wht` alongside the
inverse group and read it at 1.16x and 1.16x against the 1.17x and 1.18x above,
which is the check that the two draws are comparable. Both rows are the state
*before* the dispatch change they settled, and are the measurement rather than
the current arms: `av1_simd::iwht4x4` now returns `None` on x86_64, so a re-take
will read them the way `av1_encode_stage_wht` reads here.

#### Reading the rows

Not one row's `Best` arm is below parity, except the two the footnote above
marks as a pre-change measurement. The lowest cells anywhere in the table
are 0.98x and 0.99x, and four of the five belong to groups whose arms are the
same code: `av1_decode_frame`, `av1_encode_stage_bitstream`,
`av1_encode_stage_symbol` and `hevc_encode_cabac` have no vector kernel, so
their columns differ only by measurement noise, exactly as the aarch64
discussion above describes. The same holds for `av1_entropy_symbol`,
`av1_intra_directional`, `hevc_cabac`, `hevc_encode_bitwriter`,
`hevc_encode_cabac_bypass` and both `pcm_write` rows, which land on `1.00x` from
the same cause. The fifth is `av1_intra_paeth`'s `sse4.1` arm at 0.98x, which
does have a kernel; its `avx2` arm reads 1.05x on the same row, and the aarch64
table records the same group walking from 0.78x to 0.98x across three draws with
no code change, so this is the near-parity band that discussion is about rather
than a kernel to act on.

Two rows read at parity for a reason worth stating rather than as noise:

- `av1_intra_smooth` is `1.00x` under both vector arms because #337 removed the
  placeholder `smooth_row_{sse41,avx2}` arms. The §7.11.2.6 smooth predictor has
  no vector kernel; those arms only forwarded to `smooth_row_scalar`, and on
  x86_64 a `#[target_feature]` wrapper cannot be inlined into a row loop that
  does not carry the feature, so the forwarding cost a call the aarch64 build
  never paid. That is what the old table's 0.48x was. All three arms now call
  the reference directly, and the row is flat until a real kernel earns the arms
  back.
- `av1_encode_stage_wht` at 1.17x is not a vector win either. #337 routes
  `av1_simd::fwht4x4` to `None` on x86_64 (see #342), because the 4x4 WHT is
  fourteen SSE2-baseline adds, subtracts and shifts that LLVM already
  auto-vectorizes out of `av1_encoder::wht`, against three `transpose4`s of
  shuffle micro-operations the hand kernel adds on top. So all three arms
  execute the same scalar transform, and the ratio is only the input-limit scan
  that the x86_64 early return skips before the fallback — a few percent of a
  very small kernel, not a kernel difference. `neon` keeps the kernel and its
  2.72x, where the shuffle issue width is what makes it win.
- `av1_encode_stage_iwht` at 0.83x and 0.89x is the other direction of that same
  family, and #342 measured it rather than inferring it: the forward group could
  not settle it, because the forward pass runs three `transpose4`s where the
  inverse runs two, so the shuffle pressure that put `av1_encode_stage_wht`
  under parity is not this kernel's shuffle pressure. It turns out to be enough
  anyway. Two `transpose4`s are sixteen shuffle micro-operations contending for
  one or two ports, against a scalar loop with none, and the row reads the same
  0.83x / 0.89x pair at 320x180 and at 1080p — a property of the kernel, not of
  the frame size or of a noisy round. `av1_simd::iwht4x4` therefore joins
  `fwht4x4` on the scalar reference on x86_64 and keeps its kernel on `neon`,
  and this is now a measured dispatch on both sides of the family.

The rows with real vector work are now the ones with the largest ratios, which
is what the old table could not show. `hevc_encode_*_rgba_to_yuv420` leads at
6.27x and 6.06x, followed by `av1_deblock` at 6.28x, `av1_forward_dct_4x4` at
5.55x, `av1_deblock_boundary` at 4.90x and `hevc_color_convert` at 4.76x. The
AV1 forward transforms sit between 3.1x and 3.5x, `av1_self_guided` at 3.49x and
`av1_cdef` at 2.86x, and the motion-compensation family between 2.3x and 2.7x.

**`hevc_color_convert` moved from `1.00x` to `4.76x` because #222 landed in
between.** Every other row is attributable to #337, and this one is the move
#351 recorded without a cause, because #337 touched only `src/av1_simd` and
never `src/hevc/color_convert.rs`. The cause is neither #337 nor the harness:
the `1.00x` was a correct reading of a group that had no vector arms yet.

At `e115506f8bf6` there is no `src/hevc/color_convert.rs` in the tree at all.
The conversion is a per-pixel scalar double loop inside `picture_to_rgba` in
`src/hevc/mod.rs`, with no `simd` dispatch of any kind, so `scalar`, `sse4.1`
and `avx2` ran byte-identical code and `1.00x / 1.00x` is exactly what they
should have read. `benches/hevc_decode.rs` said as much at that commit: its
per-stage table listed the group's `Vectorized` column as "no, today". The
`convert_row_{sse41,avx2}` kernels arrived with #222 (`f695a1a`), which the
checkpoint the draw was taken on does not contain — see the paragraph on
`e115506f8bf6`'s merge base above. So the 4.76x is #222's win, showing up in the
first table drawn after it, and nothing between the two draws changed what the
group *measures*: what changed is that there is now something to measure.

This also settles the aarch64 side. That table's [sub-parity
discussion](#reading-the-sub-parity-rows) named `hevc_color_convert` as a group
whose arms are the same code; that was true of the draw it describes and is no
longer true of the crate. Its `1.27x` row is stale for the same reason and the
note there now says so.

**What keeps the hole from reopening.** A per-ISA group is only measuring its
arms if the code under it reaches a dispatch site that `zvidlib::simd` drives,
and that is now checked rather than assumed. `src/hevc/color_convert.rs` is
registered as the `hevc_color_convert` site in `simd::active_by_site`, and four
tests in `src/simd.rs` hold it there: `pinning_scalar_reaches_every_dispatch_site`
and `clearing_the_override_restores_per_site_detection` assert one selector per
site against a hand-written list that must equal `active_by_site` exactly, so a
site added without a check fails the assertion instead of going unnoticed;
`the_documented_site_table_lists_every_dispatch_site` reads the site table back
out of the `active_by_site` rustdoc and compares it; and
`every_site_reports_the_pinned_instruction_set` pins each entry of
`simd::available()` in turn and requires every site to follow. A stage whose
arms are all the same code therefore cannot be one that is registered as a site
and passing those tests, so "this group reads `1.00x`" and "this group has no
kernel" can be told apart by asking `active_by_site` rather than by reading the
prose. What is *not* checked by anything is the `Vectorized` column in [The HEVC
per-stage groups](#the-hevc-per-stage-groups) or a committed baseline table's
agreement with the kernels in force at the commit it was drawn at — both are
prose, and both are what went stale here. Quoting an old table's ratio is only
safe alongside the commit stamped on it, which is why the stamps are there.

The whole-frame encoder groups are the practical consequence.
`av1_encode_frame_q32` reads 1.49x and `av1_encode_frame_q160` 1.53x here,
against 0.52x and 0.48x in the table this one replaces: an x86_64 user of the
AV1 encoder stops paying roughly twice over for having a vector path and starts
getting about 1.5x back for it. The `_1080p` variants agree to two decimal
places, so the ratio is a property of the kernels rather than of the frame size.

`sse4.1` beats `avx2` on a minority of rows, and by enough on two of them to be
more than noise: `av1_encode_stage_coeff_ctx` was 3.04x under `sse4.1` against
2.50x under `avx2`, and the `rdo_inter` pair is 1.63x/1.64x against 1.55x. The
`Best` column already recorded `sse4.1` for these, but the dispatch site
preferred `avx2` when the host had it, so a real encode took the slower arm.
#362 answers why, and the answer is the same one for both rows: **the wide arm
never reaches its width on the block shapes these workloads actually use.** Not
lane-crossing, not downclocking, not the context gather. The detail differs.

- `av1_encode_stage_coeff_ctx`. `src/av1_simd/coeff.rs` steps along a *row* of
  the transform block, and a row shorter than the vector cannot be split across
  more than one iteration however wide the vector is. A 4x4 block is one
  iteration per row under `sse4.1` *and* under `avx2`, four of AVX2's eight
  lanes idle in every one of them, so the wide arm does identical work at twice
  the width — at best a tie. It is not even a tie, because the tail store is the
  one thing the widths do not share: `I32x::store_masked` forwards a full vector
  straight to the store instruction and stages a partial one through a stack
  buffer, and `count` is `min(size, LANES)`. At size 4 SSE4.1's four lanes are
  exactly full and take the native store, while AVX2 is partial and pays a
  32-byte spill plus a 16-byte copy on *every* store, twice a row, for
  `base_out` and `br_out` alike. The group reproduces it so cleanly because
  `coeff_context_plane` derives contexts for 4x4 blocks and nothing else. #362
  routes blocks narrower than eight lanes to the SSE4.1 kernel at the
  `av1_coeff_ctx` dispatch site, the way #342 routed `fwht4x4`, and keeps AVX2
  from size 8 up where it halves the iterations per row and stores whole
  vectors. The `avx2` column of this group was consequently the SSE4.1 kernel's
  number for as long as that routing stood, in the same sense
  `av1_encode_stage_wht`'s three columns are all the scalar transform. Giving
  AVX2 real work at size 4 needs a kernel that steps two rows at a time rather
  than one — the outputs of adjacent rows are already contiguous, so eight lanes
  are there to be filled — and #371 wrote it: `coeff::block_contexts_row_pairs`
  puts row `r` in the vector's low half and row `r + 1` in its high half, which
  makes the iteration full-width and the store native, and the site routes size
  4 back to `avx2` on the strength of the measurement below. The redirect
  survives only for sizes 1 to 3 and 5 to 7, which no vector width fits and the
  encoder never codes.

#### The #362 re-measurement

The repair is measured, not asserted, but it is deliberately recorded here
rather than folded into the table above. The `workflow_dispatch` round that
measured it landed on an **Intel(R) Xeon(R) 6973P-C (Linux/X64)** — one of the
three CPU models the table's draw explicitly measured and discarded for not
being the AMD EPYC 7763 the other rounds shared — and it is one round, not the
elementwise minimum of three. Its absolute times are therefore not comparable
with the table's, and merging two rows of it into a table attributed to a named
host would make that table attributable to no host at all, which is the failure
mode the six-round selection above exists to avoid.

What *is* comparable is the `sse4.1`-against-`avx2` sign within the round, which
is the whole claim. Measured at `539dad3d61cb` with `ZVIDLIB_BENCH_LARGE=1`,
`# host instruction sets: scalar, sse4.1, avx2`, `# dispatch site
av1_coeff_ctx: avx2`:

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `av1_encode_stage_coeff_ctx` | 3.483 ms | 930.916 µs (3.74x) | 929.755 µs (3.75x) | 3.75x `avx2` |
| `av1_encode_stage_coeff_ctx_1080p` | 31.480 ms | 8.801 ms (3.58x) | 8.568 ms (3.67x) | 3.67x `avx2` |
| `av1_encode_stage_tile` | 15.510 ms | 12.541 ms (1.24x) | 12.522 ms (1.24x) | 1.24x `avx2` |
| `av1_encode_stage_tile_1080p` | 149.547 ms | 116.654 ms (1.28x) | 116.531 ms (1.28x) | 1.28x `avx2` |
| `hevc_encode_640x352_rdo_inter` | 58.661 ms | 34.657 ms (1.69x) | 37.999 ms (1.54x) | 1.69x `sse4.1` |
| `hevc_encode_1920x1088_rdo_inter` | 554.943 ms | 328.088 ms (1.69x) | 358.684 ms (1.55x) | 1.69x `sse4.1` |

The 20% gap between the two vector arms of `av1_encode_stage_coeff_ctx` is
gone: they now read 930.916 µs and 929.755 µs, 0.13% apart, which is what
"both arms run the same kernel" looks like — the `avx2` column is the SSE4.1
kernel reached through the redirect, exactly as `av1_encode_stage_wht`'s three
columns are all the same scalar transform. `av1_encode_stage_tile`, which
contains the derivation behind the serial range coder, loses its 1.18x-against-
1.17x split the same way. The dispatch site no longer takes the slower arm, and
the `Best` column stops disagreeing with what a real x86_64 encode does.

The `rdo_inter` pair is in the table above as the control, and it is unmoved:
1.69x under `sse4.1` against 1.54x/1.55x under `avx2`, the same shape #351
recorded on a different host. Nothing in #362 touches `rdcost`, and the second
bullet above is why it would not have helped if it did.
#### The #371 re-measurement

#362's repair routed around the idle lanes; #371 removes them, and the
acceptance criterion was that the wide arm has to *win* on its own numbers
before the dispatch site takes it back. It does, and by more than the margin
#362 measured against it.

Measured at `f7b709ee62d7` — the branch's implementation commit — on an **AMD
EPYC 9V74 80-Core Processor (Linux/X64)**, one `workflow_dispatch` round with
`ZVIDLIB_BENCH_LARGE=1`, `# host instruction sets: scalar, sse4.1, avx2` and
`# dispatch site av1_coeff_ctx: avx2`. It is one round on one more model, so
these absolute times are not comparable with the table's EPYC 7763 draw or with
#362's Xeon 6973P-C round either; what is comparable, and what the criterion
turns on, is the `sse4.1`-against-`avx2` sign *within* the round.

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `av1_encode_stage_coeff_ctx` | 3.505 ms | 1.165 ms (3.01x) | 937.520 µs (3.74x) | 3.74x `avx2` |
| `av1_encode_stage_coeff_ctx_1080p` | 32.132 ms | 10.738 ms (2.99x) | 8.643 ms (3.72x) | 3.72x `avx2` |
| `av1_encode_stage_tile` | 15.320 ms | 12.544 ms (1.22x) | 12.452 ms (1.23x) | 1.23x `avx2` |
| `av1_encode_stage_tile_1080p` | 141.740 ms | 116.690 ms (1.21x) | 115.490 ms (1.23x) | 1.23x `avx2` |
| `hevc_encode_640x352_rdo_inter` | 73.844 ms | 45.439 ms (1.63x) | 47.741 ms (1.55x) | 1.63x `sse4.1` |
| `hevc_encode_1920x1088_rdo_inter` | 699.730 ms | 431.420 ms (1.62x) | 451.480 ms (1.55x) | 1.62x `sse4.1` |

The two vector arms of `av1_encode_stage_coeff_ctx` are 24% apart and the wide
one is ahead: 937.520 µs against 1.165 ms at 320x180, 8.643 ms against 10.738 ms
at 1080p, the same ratio at both sizes because the group derives contexts for
4x4 blocks and nothing else. Under #362's routing those two columns read the
same number to within 0.13%, because they *were* the same kernel; the split
reopening in AVX2's favour is what a kernel that actually fills its lanes looks
like. `av1_encode_stage_tile`, which runs the derivation behind the serial range
coder, carries a smaller share of it through — 1.23x against 1.22x — as it
should, since the context pass is one stage of that group rather than all of it.

The `rdo_inter` pair is the control, unchanged by anything here: 1.62x/1.63x
under `sse4.1` against 1.55x under `avx2`, the same shape #351 and #362 both
recorded on other hosts. Nothing in #371 touches `rdcost`, and this round
predates #370's own repair — that row is measured below.

#### The #370 re-measurement

Unlike the round above, this one landed on the **AMD EPYC 7763 64-Core
Processor (Linux/X64)** — the same host the committed x86_64 table was measured
on — so its columns can be read against that table directly. It is still one
round rather than the elementwise minimum of three, which is why it is recorded
here rather than merged into the table; the controls below are what carry the
attribution. Measured at `6213a5580b78` with `ZVIDLIB_BENCH_LARGE=1`,
`# host instruction sets: scalar, sse4.1, avx2`, `# dispatch site
hevc_rdcost: avx2` ([run
33615242194](https://github.com/lsegal/zvidlib/actions/runs/33615242194)):

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `hevc_encode_640x352_rdo_inter` | 88.922 ms | 55.341 ms (1.61x) | 55.886 ms (1.59x) | 1.61x `sse4.1` |
| `hevc_encode_1920x1088_rdo_inter` | 843.440 ms | 523.280 ms (1.61x) | 530.010 ms (1.59x) | 1.61x `sse4.1` |
| `hevc_encode_640x352` | 101.710 ms | 65.083 ms (1.56x) | 65.257 ms (1.56x) | 1.56x `sse4.1` |
| `hevc_encode_1920x1088` | 974.890 ms | 624.680 ms (1.56x) | 628.680 ms (1.55x) | 1.56x `sse4.1` |
| `hevc_encode_640x352_rdo_intra` | 5.591 ms | 3.765 ms (1.48x) | 3.721 ms (1.50x) | 1.50x `avx2` |
| `hevc_encode_1920x1088_rdo_intra` | 52.104 ms | 34.900 ms (1.49x) | 34.693 ms (1.50x) | 1.50x `avx2` |

The `avx2` column of the `rdo_inter` pair is what moved, and only it: 57.659 ms
to 55.886 ms and 547.481 ms to 530.010 ms against the table above, both about
3% faster, while the same rows' `scalar` and `sse4.1` columns land within 1% of
their table values (88.922 against 89.626, 55.341 against 54.769, 843.440
against 851.069, 523.280 against 520.861). `rdo_intra`, which runs the same
`rdcost::satd` through a mode search that never calls `sad`, is unmoved at
1.48x/1.50x against the table's 1.49x/1.49x — one round of run-to-run noise on
a row the change does not reach. So the 3% is attributable to the routing
rather than to the host or the round.

The gap between the two vector arms goes from 5.3% and 5.1% to 1.0% and 1.3%,
and `Best` still reads `sse4.1` — this repair stops the `avx2` arm paying for a
width it never gets, which brings it level, and level is the whole of what
routing can buy. The residual 1% is real and expected: a CTB of 16 keeps `satd`
on the genuine AVX2 pair loop at `w == 16`, and these groups are whole-frame
encodes with the other AVX2 dispatch sites (`hevc_fwd_transform_quant`,
`hevc_recon`, `hevc_prediction_filters`) still live in both arms. Making the
`avx2` arm actually faster than `sse4.1` here needs the batched search of #387,
not a threshold.

The whole-frame groups are the practical consequence. `hevc_encode_640x352`
reads 65.257 ms under `avx2` against the table's 67.101 ms and
`hevc_encode_1920x1088` 628.680 ms against 646.477 ms, so an x86_64 user
encoding HEVC on an AVX2 host gets about 2.8% of a whole encode back — the two
arms are now 0.3% and 0.6% apart where the table has them 4.3% and 4.0% apart.
`av1_encode_stage_coeff_ctx` reads 1.4647 ms and 1.4745 ms on this host, 0.7%
apart, which is #362's redirect reproducing on the table's own hardware.

- `hevc_encode_*_rdo_inter`. The same family, a different mechanism, and *not*
  the same fix. `rdcost::sad_avx2`'s 256-bit loop needs `w >= 32` and
  `satd_avx2`'s vector pair needs `w >= 16`, but `rdo.rs` searches a `CTB` of 16
  and its candidate partitions subdivide that, so the 32-wide SAD branch is
  never taken at all and every sub-partition of width 8 or 4 falls through to
  `satd_8x8_sse41`. Both AVX2 metrics therefore execute the same 128-bit body as
  their SSE4.1 counterparts, plus a 256-bit zero, a `vextracti128` fold and the
  `vzeroupper` an AVX2 `#[target_feature]` function emits on return — a fixed
  per-call cost, on a motion search that issues 81 SAD calls per candidate, which
  is exactly the loop where a per-call cost cannot amortize. So it is a
  per-call setup the wider step does not pay for rather than idle lanes and a
  staged store, and the repair is to widen what the search hands the kernel (or
  to route these two the way `coeff_ctx` is now routed) rather than anything
  #362 changes. #370 carried it, and took the second of those two: `sad` routes
  blocks narrower than 32 and `satd` blocks narrower than 16 to the SSE4.1
  kernel, each at its own AVX2 body's threshold rather than at one shared
  number. Widening what the search hands the kernel is the other repair and is
  still open as #387 — it is the one that would make AVX2 *win* here rather than
  stop losing, but it moves the search's candidate ordering and early
  termination with it, so it is an optimization rather than a defect fix.
  Measured under [The #370 re-measurement](#the-370-re-measurement).

## Hardware HEVC decoders

`benches/hevc_hardware.rs` is its own `[[bench]]` target. It
measures whichever platform fixed-function HEVC decoder the host provides —
NVDEC, Windows Media Foundation, or VideoToolbox — against the pure-Rust
software decoder on the bundled 1080p sample.

```sh
cargo bench --bench hevc_hardware                     # hardware arms only
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench hevc_hardware  # plus the software baseline
```

Three things make this target different from the rest of the suite:

- **No scalar-vs-SIMD arms.** These are opaque drivers and OS frameworks;
  `zvidlib::simd`'s process-wide override does not reach an instruction they
  execute, so scalar and vector arms would differ only by noise. The group name
  also carries no `simd=on`/`simd=off` build tag, since the hardware numbers are
  identical in both builds. The software baseline group does carry it.
- **Setup latency is a separate benchmark from throughput.** A backend pays a
  real one-time cost — a CUDA context and parser, an MFT and its D3D11 device, a
  VideoToolbox decompression session — and averaging it into a throughput figure
  misrepresents both. `<arm>/session_setup_to_first_frame` times construction
  through the first delivered frame; `<arm>/steady_state` starts its clock only
  after that frame is out. Both use `Bencher::iter_custom` to draw the line.
- **It skips, it does not fail.** With no hardware decoder the group prints why
  and returns, so `cargo bench` works on a dev box without one — the same policy
  as the `#[ignore]`d `tests/native_hevc_hardware.rs`.

The software baseline sits behind `ZVIDLIB_BENCH_LARGE=1` like every other group
that puts the 1080p sample through the software decoder. Both arms decode the
same 32-frame window, which is what makes their ratio a ratio: the sample's
frames are not equally expensive (a key frame costs far more than the
hierarchical B-frames after it), so arms measured over different frame counts
would be comparing different work. The run prints the ratio directly.

The setup arm reports the *warm* per-session cost, since criterion builds a
session per iteration after the framework has already initialized. The single
untimed pass printed above the criterion output reports the cold one, which
includes one-time driver/framework initialization; a caller pays that once and
the warm cost on every seek-driven reset.

### Measured backends

One row per measurement run, naming the host it was taken on. `Steady state` and
`Warm setup` are the criterion `<arm>/steady_state` and
`<arm>/session_setup_to_first_frame` figures; `Cold setup` is the single untimed
pass. The software column is the `ZVIDLIB_BENCH_LARGE=1` baseline arm from the
*same* run, which is what makes the ratio a ratio rather than a comparison
across hosts.

| Backend | Host | Steady state | Warm setup | Cold setup | Software, same host | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| VideoToolbox | idle Apple Silicon (#170) | 167 Mpx/s | not recorded | not recorded | 15 Mpx/s | ~11x |
| VideoToolbox | Apple M1, macOS 26.5 (#282) | 163 Mpx/s, 78.6 fps | 16.7 ms | 114 ms | 38.5 Mpx/s, 18.6 fps | ~4x |
| NVDEC | `ubuntu-latest`, Azure VM, x86_64 (#282) | not measured | — | — | — | — |
| Media Foundation | `windows-latest`, Hyper-V Video adapter (#282) | not measured | — | — | — | — |

Read the ratio as an order of magnitude, not a two-digit figure. The hardware
arm is stable run to run — a fixed-function block decoding a fixed window — while
the software arm is a long single-threaded workload and varies several-fold on a
loaded host, so the ratio moves with the host's other work rather than with
anything the decoders did. The two VideoToolbox rows are exactly that: their
hardware numbers agree to within a few percent and their software numbers differ
by 2.5x, which is where the whole gap between ~11x and ~4x lives. Neither row is
the wrong one; the ratio is a property of the host as much as of the decoders.

The #282 row is the minimum of two back-to-back runs on the same host rather
than either run's own reading. The second run came out 40% slower on the
*hardware* arm — the arm this file calls stable — which is the host announcing
contention rather than anything the decoder did, so the slower run is discarded
on that evidence instead of averaged in.

### Backends that could not be measured

NVDEC and Media Foundation both have code in the tree
(`src/hevc/nvdec.rs`, `src/hevc/windows_mf.rs`) and both compile and run the
benchmark, but no host with the fixed-function hardware behind either one was
available. The bench needed no changes to reach that conclusion on either
platform: it built and skipped cleanly, exactly as designed.

- **NVDEC**, on `ubuntu-latest`: no NVIDIA GPU and no driver. The probe reports
  `NVDEC: NVIDIA CUDA driver is unavailable: libcuda.so.1: cannot open shared
  object file`. GitHub's standard hosted Linux runners are Azure VMs with no
  attached GPU — `nvidia-smi` is absent and neither `libcuda.so.1` nor
  `libnvcuvid.so.1` is on the loader path — so no configuration of a standard
  runner reaches this backend. Measuring it needs a self-hosted or GPU-class
  runner, or a physical NVIDIA host.
- **Media Foundation**, on `windows-latest`: no D3D11 video device. The probe
  reports `Media Foundation: D3D11 video decode is unavailable: No such
  interface supported (0x80004002)` — the runner's only display adapter is
  `Microsoft Hyper-V Video`, a paravirtualized adapter that exposes no
  `ID3D11VideoDevice`, so the `D3D_DRIVER_TYPE_HARDWARE` device
  `windows_mf::is_available` requires cannot be created. `CLSID_MSH265DecoderMFT`
  is not registered on the image either, so even a software MFT fallback is
  absent. Measuring it needs a Windows host with a real GPU.
- The same Windows run also found NVDEC unavailable there
  (`NVIDIA CUDA driver is unavailable: LoadLibraryExW failed`; `nvcuda.dll` and
  `nvcuvid.dll` are both absent from the image), so neither of that platform's
  two candidate backends is reachable on a hosted runner.

Both numbers stay open until a host with the hardware runs the benchmark. The
Windows run is worth one note of its own beyond the missing row: it is the first
time the crate has been built and run on Windows in CI at all — `.github/workflows/ci.yml`
has only ever had Linux jobs — and the `windows` and `libloading` target
dependencies, the Media Foundation backend, and the benchmark suite all compiled
without a warning.

### Frame readback

`VideoDecoder` hands back a host-side `VideoFrame`, the HEVC decoder
configuration only accepts `PixelFormat::Rgba8`, and each backend maps its own
surface and converts to RGBA inside `submit` — so a caller sees the
fixed-function decode and the host round trip as one number. For a playback
pipeline the round trip is often the part that bounds throughput, which is what
made it worth separating (`#151`, `#170`, `#283`).

`hevc_hardware_readback` is the group that separates it, and it runs whenever
the hardware arm does:

| Benchmark | What it measures |
| --- | --- |
| `hardware/surface_copy` | making the decoded surface CPU-readable: `cuvidMapVideoFrame` + `cuMemcpyDtoH` (NVDEC), the staging-texture `CopySubresourceRegion` + `Map` (Media Foundation), or `CVPixelBufferLockBaseAddress` (VideoToolbox) |
| `hardware/color_convert` | the NV12-to-RGBA pass over those bytes and the RGBA allocation it fills |

The two are split because they scale differently by host: the surface copy is a
PCIe transfer on a discrete GPU and little more than a lock on unified memory,
while the conversion is host CPU work everywhere. The run also prints both as
ms/frame and as a percentage of the `steady_state` decode they are part of, which
is the ratio the issues above asked for.

These are attribution numbers, not wall-clock ones. Each backend charges its own
per-frame phases to `zvidlib::hevc_hardware_readback`, and one criterion
iteration decodes the same 32-frame window `steady_state` does and reports only
the nanoseconds that window spent in the phase under test — so an iteration's
wall time is longer than the number it prints, by exactly the decode it had to
run to produce it. The seam counts frames as well as nanoseconds, and the group
asserts the count matches the window, so a backend that stopped reporting reads
as a failed run rather than as free readback.

One host has run it so far, and each row names its own:

| Host | Backend | `surface_copy` | `color_convert` | Share of `steady_state` |
| --- | --- | --- | --- | --- |
| Apple Silicon (unified memory) | VideoToolbox | ~3 us/frame | ~10 ms/frame | roughly two thirds to three quarters of 13-15 ms/frame |
| discrete NVIDIA GPU | NVDEC | not yet measured (`#318`) | not yet measured | — |
| Windows + D3D11 | Media Foundation | not yet measured (`#318`) | not yet measured | — |

The Apple Silicon numbers are over the same 32-frame window `steady_state` uses,
and its `steady_state` figure moves with the host's other work. The split is the
useful part of that: on unified memory there is no transfer to remove — the
`surface_copy` phase there is a `CVPixelBufferLockBaseAddress` and not a copy at
all — and the host round trip is almost entirely the crate's own NV12-to-RGBA
pass, the same conversion that is the largest single item in a *software*
decode.

That is one host's answer and not the general one. A discrete-GPU host is
expected to read differently, with a real PCIe transfer in `surface_copy`
(`cuvidMapVideoFrame` plus `cuMemcpyDtoH`, or the staging-texture
`CopySubresourceRegion` plus `Map`) rather than a lock — which is the case the
split was built to expose. `#300` corrected the stale pointer that used to stand
here, and `#318` carries the measurement itself; it needs a host with the
hardware, for the same reason the [hardware decoder
table](#hardware-hevc-decoders) above still has empty rows.

There is no readback arm on the software baseline. The seam covers the
fixed-function backends; the software decoder's own conversion is already the
`color_convert` stage in [the decode breakdown](#where-hevc-decode-time-actually-goes)
and the `hevc_color_convert` per-stage group.

#### Why a measurement seam and not a zero-copy output path

Issue #283 asked the broader question the measurement gap implied: should the
decoder expose the decoded surface *before* readback, so a GPU-side consumer (a
texture upload, a wgpu or WebGL path) could skip the host round trip entirely?
That was decided against, for now:

- It is three platform handle types (`CVPixelBuffer`, a `CUdeviceptr` plus its
  context, an `ID3D11Texture2D` plus the device that owns it), each with its own
  lifetime and threading contract, in a public API — and the crate does not own
  the drivers or frameworks whose contracts it would be promising to keep.
- It requires a second public `PixelFormat` family (NV12), since no backend
  produces RGBA on the GPU today.
- The benchmark that motivated it does not need it. A benchmark wants the cost
  of the copy that runs, not a way to avoid it, and the seam above measures
  exactly that code rather than a reimplemented stand-in.

The third point is the one that is only known for unified memory. It rests on
the recorded ratio, where the transfer is ~3 us against ~10 ms of conversion, so
there is no round trip worth removing. A discrete-GPU host that reverses that
ratio — a PCIe transfer dominating the conversion — would not settle the first
two objections, but it would remove the third, and this decision should be
re-read against that number rather than against the Apple Silicon one when
`#318` produces it.

The zero-copy path stays unbuilt until a caller needs it; the case for it would
be a real GPU-side consumer, not a measurement. Until then
`zvidlib::hevc_hardware_readback` is `#[doc(hidden)]` and unstable, like
`hevc_decoder_bench` and `hevc_decode_profile`, and its per-frame instrumentation
is unconditional for the same reason theirs is: a feature-gated profiler measures
a build nobody ships. It costs two `Instant::now()` reads and a relaxed
`fetch_add` per phase per frame. Its accumulators are process-wide atomics rather
than thread-locals because NVDEC and VideoToolbox deliver frames from a callback
that need not run on the submitting thread.

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
| `..._reconstruct` | encode-side reconstruction (predict + add residual per coded block, through `hevc_recon`) plus the §8.7.3 SAO parameter search and the §8.7.2 deblocking filter and §8.7.3 SAO over the reconstructed picture |
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

The encoder has four SIMD dispatch families of its own: `hevc_rdcost`, the SAD
and SATD distortion metrics the mode search calls; `hevc_fwd_transform_quant`,
the forward transform and quantization; `hevc_recon`, the §8.6.6 reconstruction
loop and the encode-side §8.7.3 SAO parameter search; and `hevc_colorconv`, the
RGBA8 to YUV420 input conversion. `..._reconstruct` reaches `hevc_recon` and,
after it, the decoder's already-vectorized deblocking and SAO filter kernels.

`..._reconstruct` only separated across instruction sets once `hevc_recon`
existed. Before it, the group's arms barely moved — the in-loop filter kernels
it called were a minority of its cost, while the reconstruction loop and the SAO
parameter search in front of them were scalar. Measured on a contended Apple
Silicon host, best of three interleaved rounds (a floor; read the ratio rather
than the absolute time): 640x352 29.2 ms scalar against 11.0 ms NEON, and
1920x1088 121.4 ms scalar against 38.9 ms NEON, where before it read 11.2 ms
against 9.6 ms at 640x352 and did not separate at all at 1080p.

The SAO parameter search's band-offset half is *not* part of that separation
on any instruction set, and that is a measured result rather than a gap - on
x86_64 it is a measured result that took two different benchmarks to reach.
`band_offset_row` is a `hevc_recon` dispatch site whose 32-way scatter is not
expressible in SSE4.1, AVX2 or NEON, so the only vectorizable work is the
clamp, shift and widened subtraction in front of it.

On NEON that is not enough even in isolation. Measured on the same contended
Apple Silicon host against the scalar reference over L1-resident runs of 16 to
1024 samples, best of interleaved rounds, and measured again from a standalone
harness: staging the classification into buffers and then scattering them read
0.42-1.30x and scattering straight out of the vector lanes read 0.44-1.24x.
Neither separates from scalar - both straddle 1.00x by less than the spread
between repeats of the same measurement, which is what this host's contention
looks like. This group agreed: its NEON arm did not improve.

On x86_64 the classification *does* pay in isolation, and the argument that it
would not - that masking 32 accumulators costs more than the read-modify-writes
it replaces - was an analytical one carried over from the NEON measurement.
`bench_band_offset_row` timed it across five CPU models (nine `ubuntu-latest`
draws plus two `macos-15-intel`), best of nine interleaved rounds per draw.
**Group the draws by CPU model.** `ubuntu-latest` is drawn from several models
and they disagree by more than the effect: the same AVX2 arm read 1.10x, 1.45x
and 0.97x on the first three draws purely because they landed on three
different CPUs. Within a model the ratio reproduces to about +/-0.02 across
independent draws.

Lane-scatter shape against the scalar reference, by run length:

| CPU model | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| Intel Xeon 6973P-C | 1.16x | 1.53x | 1.53x | 1.51x |
| Intel Xeon Platinum 8573C | 1.16x | 1.28x | 1.44x | 1.39x |
| Intel Core i7-8700B | 1.11x | 1.25x | 1.28x | 1.30x |
| AMD EPYC 7763 | 1.24x | 1.13x | 1.10x | 1.10x |
| AMD EPYC 9V74 | 1.23x | 0.97-1.02x | 1.01x | 0.97x |

Four of the five models separate at every length, and the SSE4.1 shape is ahead
of scalar on every model at every length (1.02-1.45x). **On that harness alone
the kernel is worth landing. This group says it is not.**

The whole-picture comparison is the one the decision was taken on, and it was
run as a paired branch-against-base measurement rather than against a recorded
figure: both trees built and timed on the same host, interleaved within a
round, five rounds per draw, twelve draws across five models. The group's own
`scalar` arm is the control - it resolves to the same scalar reference in both
trees, so it has to read 1.00x, and it does to within +/-0.02 everywhere.

| CPU model | draws | `avx2` | `sse4.1` | `scalar` (control) |
|---|---|---|---|---|
| Intel Xeon Platinum 8573C | 1 | 1.00x | 1.01x | 1.01x |
| Intel Xeon Platinum 8370C | 2 | 1.01x | 1.00-1.02x | 1.00x |
| Intel Core i7-8700B | 1 | 0.99-1.00x | 1.01-1.02x | 1.00-1.02x |
| AMD EPYC 7763 | 5 | 0.97-0.99x | 0.95-0.98x | 0.99-1.00x |
| AMD EPYC 9V74 | 3 | 0.94-0.95x | 0.95x | 1.00x |

On the Intel parts the kernel is invisible against its own control; on both AMD
parts it is a 2% to 6% regression, reproducing across independent draws with
every round signed the same way and well outside what the control moves by. The
two harnesses measure different things and the encoder's is the one that
counts: `bench_band_offset_row` calls the kernel back-to-back over one
L1-resident run of up to 1024 samples with `stats` hot and the call fully
predicted, while the encoder calls it once per CTB row - 16 to 64 samples - in
between the rest of reconstruction. **No x86_64 kernel is dispatched to.** Both
x86 shapes stay `#[cfg(test)]` as the measurement apparatus, asserted bit-exact
so the figures above are figures for kernels that would be correct to land.

If you re-time this, re-time it the same way: a branch-against-base ratio taken
on one host with the arms interleaved, with the `scalar` arm read as a control.
A recorded absolute figure from another draw is not a comparison - the four
models above differ by more than the effect.

**AVX-512 was timed and does not separate even in isolation.**
`ubuntu-latest` draws AVX-512CD hosts, so the `vpconflictd` shape #305 pointed
at was reachable. Resolving the scatter's duplicate indices inside the vector
unit reads **0.42-0.56x** on every host that can run it: the conflict-resolving
pointer chase costs more than the read-modify-writes it removes, at 32 bands
over 16 lanes where duplicates are rare. Merely widening the classification to
512 bits reads 1.14-1.52x on the Intel parts but **0.21-0.30x** on Zen 5, whose
double-pumped AVX-512 makes the wider classification a large loss.

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

`..._reconstruct` separated across instruction sets once `hevc_recon` existed.
Measured on a contended Apple Silicon host, best of three interleaved rounds
(treat these as a floor, and read the ratio rather than the absolute time):
640x352 29.2 ms scalar against 11.0 ms NEON, and 1920x1088 121.4 ms scalar
against 38.9 ms NEON. Before it, the same group read 11.2 ms scalar against
9.6 ms NEON at 640x352 and did not separate at all at 1080p, because the in-loop
filter kernels it called were a minority of its cost.

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
