# Benchmarks

zvidlib's benchmarks run under [criterion](https://docs.rs/criterion) with
`harness = false`, across nine bench targets that share `benches/support/`:

| Target | Measures |
| --- | --- |
| `benches/codec.rs` | codec work: decode, encoder inputs, and the per-ISA SIMD groups |
| `benches/av1_decode.rs` | the AV1 software decoder: whole-frame decode and every hot stage, scalar versus SIMD |
| `benches/av1_encode.rs` | the native AV1 encoder: whole-frame encode, every stage, and the forward-transform kernels, scalar versus SIMD |
| `benches/audio_decode.rs` | the audio decode path: AAC access units and `AacSampleReader` range/seek reads |
| `benches/audio_mux.rs` | the audio container path: MP4 muxing, sample-table growth, demux, and gapless timing |
| `benches/hevc_encode.rs` | the pure-Rust HEVC encoder, whole-frame and per-stage |
| `benches/hevc_decode.rs` | the HEVC software decoder: whole-frame decode and every hot stage, scalar versus SIMD |
| `benches/hevc_hardware.rs` | the platform fixed-function HEVC decoders against the software one |
| `benches/exact_seek.rs` | what an exact frame at an arbitrary point costs, by backend and by random-access cadence |

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
cargo bench --bench hevc_decode   # the HEVC software decoder only
cargo bench --bench hevc_hardware # the platform hardware HEVC decoders
cargo bench --bench exact_seek    # exact-seek cost by backend and cadence
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
cargo bench --bench codec -- av1_deblock_luma
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

Its whole-frame content is `support::av1_gray8_planes`, which borrows the HEVC
encoder fixture's luma, so `What the synthetic content's value distribution is,
and what it cannot answer` below applies to these groups unchanged.

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

That became issue #378, and the section below is its answer: the kernel does go
about twice as fast at 16-bit width, and it turns out that only buys anything on
one of the four phase cases, for a reason that is about the *caller* rather than
the kernel.

#### `filter_taps` at 16-bit width: where the doubled lane count survives the trip

Issue #378 asked whether narrowing the interpolation sample path from `i32` to
`i16` recovers what #309 and #312 could not. Three findings, in the order they
constrain each other.

**The 16-bit accumulator is in range at eight bits, and only at eight bits.**
`shift1` is `Min( 4, BitDepth − 8 )`, so it is zero for eight-bit content and the
tap accumulation is not scaled down before it lands. A partial sum of
`Σ coeffs[t] · taps[t][i]` taken in tap order is a subset sum, so it is bounded by
the all-negative and all-positive subsets. The widest §8.5.3.3.3.2 kernel is
`[ −1, 4, −11, 40, 40, −11, 4, −1 ]`, whose positive coefficients sum to 88 and
negative to −24, so with samples in `0..=255` the reachable interval is
`−6120..=22440` — inside `i16`, with room to spare. At nine bits it is not
(`88 · 511 = 44968`), and the arrangement collapses. So this is an eight-bit fast
path, not a change to how planes are represented, and the higher bit depths keep
the `i32` kernel untouched.

**Only an `i16` *accumulator* wins; `i16` operands alone win nothing.** This is
the finding that decides the shape of everything else. A widening multiply-
accumulate — `vmlal_s16` on NEON, `vmlal_high_s16` for the top half — reads
`i16` lanes but still produces four `i32` lanes per instruction, exactly what
`vmlaq_s32` produces. Timed against the `i32` kernel over the row-length sweep it
reads 1.18x at a row of 8, 1.04x at 64 and **1.00x at 256**: the whole of its
small-row advantage is halved load bandwidth, and at long rows there is nothing
left. The doubling only exists with `vmlaq_n_s16`, which keeps eight lanes all
the way through the accumulator. `simd::filter_taps_narrow` is that kernel, and
`simd::measure_narrow_filter_taps` A/Bs it against `filter_taps` on the same
sweep, in one process, interleaved, best of nine rounds:

| Samples per call | `i32` `neon` | `i16` `neon` | narrow | `i32` `scalar` | `i16` `scalar` | narrow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3.88 ms | 4.05 ms | 0.96x | 13.56 ms | 8.87 ms | 1.53x |
| 8 | 2.22 ms | 2.20 ms | 1.01x | 7.60 ms | 6.94 ms | 1.09x |
| 16 | 1.76 ms | 1.31 ms | 1.34x | 3.85 ms | 5.34 ms | 0.72x |
| 32 | 1.56 ms | 0.94 ms | 1.66x | 2.35 ms | 2.47 ms | 0.95x |
| 64 | 1.60 ms | 0.87 ms | 1.84x | 2.03 ms | 1.86 ms | 1.09x |
| 128 | 1.52 ms | 0.82 ms | 1.85x | 1.73 ms | 1.55 ms | 1.12x |
| 256 | 1.50 ms | 0.80 ms | 1.89x | 1.63 ms | 1.45 ms | 1.12x |

The kernel reaches **1.89x** where #309 established the `i32` kernel was already
at the M1's integer SIMD multiply throughput, which is the expected answer: the
limit was multiplies per cycle, and this issues half as many for the same work.
Below a row of eight there is no 16-bit vector loop to reach — the call is the
4-wide widening remainder plus the cost of having narrowed its source — so the
narrow-width chroma blocks keep the `i32` kernel.

**But a block is not a kernel, and only one phase case keeps the win.**
`measure_narrow_vs_wide_block` A/Bs the two arms through the whole of
`interp_block` — the block walk, the source narrowing and the allocations
included — at the half-pel phase, in one process, best of fifteen rounds, with
the two arms asserted equal sample-for-sample before anything is timed:

| Phase | 8x8 | 16x16 | 32x32 | 64x64 |
| --- | ---: | ---: | ---: | ---: |
| horizontal-only (a/b/c) | 0.82x | 0.99x | 0.94x | 1.04x |
| vertical-only (d/h/n) | 1.37x | 1.22x | 1.25x | 1.43x |
| two-dimensional (e/i/p, f/j/q, g/k/r) | 0.92x | 0.95x | 0.99x | 1.02x |

Same kernel in all three rows, at 1.89x in isolation, and only the middle row
keeps any of it. **What separates them is who pays to narrow the source.** The
vertical-only case reaches its taps through `RefPlane::gather`, which
materializes a `w x ( h + N − 1 )` buffer either way, so writing that buffer as
`i16` instead of `i32` costs nothing at all — same store count, half the bytes —
and the kernel's advantage arrives intact. The other two reach theirs through
`RefPlane::row_window`, which **borrows the plane outright with no copy** when
the window lies inside it, which is the common case. Narrowing there has to
introduce a materialized pass that the `i32` path never performs, and that pass
costs about what the wider lanes save. The two-dimensional case is worse again
for a second, independent reason: only its horizontal pass can narrow at all. Its
vertical pass multiplies a 16-bit intermediate by a coefficient of up to 58, so
it needs a 32-bit accumulator, and by the finding above that is the same lane
count the `i32` kernel already issues.

So `interp_block` takes the narrow path for the vertical-only phase, at eight
bits, on a vector backend, at rows of eight samples or more — the four conditions
`inter_pred::narrows` spells out — and nowhere else. Both arms stay spelled out
inside `interp_block_with_width` rather than one of them being deleted, the same
arrangement `measure_2d_ring_vs_flat` uses, so the table above stays reproducible
after the decision it justifies.

**The whole-decode effect is below this host's noise floor, and the null control
is what says so.** `inter_pred_filter` is about 27% of decode proper, the
vertical-only phase is 3 of the 16 Table 8-8 combinations, and a 1.2-1.4x on that
share is a low single-digit percentage of a decode. Paired against `main` through
`hevc_decode_profile`, 48 frames, twelve interleaved rounds, elementwise minimum,
the `neon` arm read 1.10x in one round set and 0.83x in another. The `scalar` arm
is the control that settles it: `narrows` is false for `Isa::Scalar`, so both
binaries execute identical code on that arm, and it still read 1.06x and 1.07x in
those same two round sets. **A ±7% disagreement on an arm where the answer is
known to be 1.00x is larger than the effect being measured**, and the host was
running other work throughout. The in-process A/Bs above are the instrument for
this question, for exactly the reason `measure_2d_ring_vs_flat` already gives:
separate benchmark processes on this host disagree with each other by more than
the effect. The committed `hevc_inter_pred` and `inter_pred_filter` rows are
therefore left as they stand rather than redrawn from measurements that cannot
resolve the change.

No decoded sample moves on any backend or at any bit depth.
`the_eight_bit_block_path_matches_the_per_sample_equations` runs every block
shape, origin and phase of the eight-bit block path against the normative
per-sample §8.5.3.3.3.2 / §8.5.3.3.3.3 equations, which are `i32` throughout —
the guard that matters here, because `every_backend_matches_scalar_luma_block`
compares vector kernels against the scalar one and at eight bits both of its
sides would be the narrow formulation.

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

## What a drag preview costs the frame under the pointer

The other profiling example measures an interaction rather than a codec.
Dragging `native_gl`'s timeline bar walks a background decoder towards the frame
under the pointer and publishes pictures on the way, so the drag keeps moving
(issue #363); `PREVIEW_INTERVAL` in `examples/native_gl/scrub.rs` is how often it
publishes. That interval was arithmetic — a dozen pictures over the 613-frame
walk issue #354 timed at 1.7 s, about 4% on top — and issue #379 is what
happened when it was measured.

`examples/scrub_preview_profile.rs` drives the same `FrameService` the window
drives, over the same bundled sample through the same hardware decoder, with no
window and no renderer in the way:

```sh
cargo run --release --features native --example scrub_preview_profile
cargo run --release --features native --example scrub_preview_profile -- 5
cargo run --release --features native --example scrub_preview_profile -- 3 80 150 400
```

The first argument is runs per arm and the rest are cadences in milliseconds.
Each arm builds its own service, so every walk starts from a cold decoder, and
each reports the fastest of its runs: the decoder is shared hardware and
anything else on the host only ever adds time. Its baseline arm is playback's
exact request, which publishes nothing on the way to its target — what the drag
did between #355 and #363 — so the overhead each cadence reports is measured
against the same decoder in the same process.

### The sweep

Walking to frame 767 of `examples/media/BigBuckBunny.mp4` from cold on an Apple
Silicon host (M1, 8 cores) through VideoToolbox, `--release`, fastest of five
runs per arm. No previews at all: **1.25 s**, one picture.

| interval | arrival | published | spacing | converted | overhead | per convert |
| --- | --- | --- | --- | --- | --- | --- |
| 80 ms | 3.52 s | 34 | 104 ms | 306 | 182% | 7.4 ms |
| 150 ms | 7.03 s | 50 | 141 ms | 765 | 463% | 7.6 ms |
| 250 ms | 5.56 s | 24 | 232 ms | 578 | 345% | 7.5 ms |
| 400 ms | 3.39 s | 12 | 282 ms | 299 | 171% | 7.2 ms |
| 600 ms | 2.38 s | 8 | 297 ms | 173 | 91% | 6.5 ms |
| 800 ms | 2.31 s | 7 | 330 ms | 156 | 85% | 6.8 ms |
| 1200 ms | 2.18 s | 6 | 363 ms | 129 | 74% | 7.2 ms |
| 1600 ms | 1.82 s | 5 | 364 ms | 100 | 43% | 5.5 ms |
| 2400 ms | 1.89 s | 5 | 378 ms | 102 | 51% | 6.3 ms |

`published` is what the service published, counted by the service rather than by
what the harness collected — a picture the render thread never draws still cost
its conversion. `converted` is what the walk converted to RGBA, which is the
larger number and the one the time goes into, and `spacing` is arrival divided
by publishes: how often the picture under the pointer actually moves, which is
what #363 asked for.

### What it says

**The 4% estimate was wrong by two orders of magnitude, and the per-picture
price was not what it got wrong.** Per converted picture the cost is 5.2 ms to
7.6 ms across the whole sweep, which is the 6.6 ms #355 measured through the
`hevc_hardware_readback` seam. What the estimate missed is the count: the
150 ms cadence converts **765** of the sample's 768 pictures on one walk, not a
dozen, and the frame under the pointer arrives in 7.03 s rather than 1.7 s —
worse than the 6.4 s issue #354 was opened for.

The count is the reader's cache tail. `ExactFrameReader::get` keeps the frames
immediately behind its target as well as the target, as many as
`Limits::max_cached_frames` holds (32 by default), which is what makes stepping
backwards free after a seek. A preview walk calls `get` once per published
picture and pays that tail every time, so a stride shorter than the tail
converts everything it passes and the walk is back to #354's behaviour by a
different route. Issue #402 tracks the tail; the cadence can only work around
it.

Working around it is what `PREVIEW_INTERVAL` now does. It is 1.6 s, the knee:
past it the walk neither arrives sooner nor publishes more — the stride is
capped by `MAXIMUM_STRIDE` and the tail is what remains — and at it the frame
under the pointer arrives in 1.82 s, #354's 1.7 s and about 8%, while the walk
still publishes five pictures at 364 ms apart. The motion #363 asked for is not
lost with it: since issue #374 a background pass keeps a shrunk picture every
half second of the track, and a drag draws one for wherever the pointer is
within a frame of the window's, so what this cadence owes #363 is the
full-resolution picture catching up rather than the movement itself.

The sweep is not monotonic, and that is the same mechanism seen from the
outside: the stride is computed from the walk's own measured per-frame rate,
that rate includes the conversions the step paid for, and a shorter stride
converts a larger fraction of what it passes. Shortening the interval lengthens
the per-frame estimate the next stride is computed from, so 80 ms and 400 ms
land on similar strides from opposite directions while 150 ms sits in the
region where every frame is converted.

## What an exact frame at an arbitrary point costs (`--bench exact_seek`)

Issue #374 asked for "<50 ms to any part of a video". #383 answered the *scrub*
half of it and left the exact half open, on the reasoning that reaching
presentation frame *n* means decoding every sample from the nearest preceding
random-access point, and the bundled sample has exactly one. Issue #395 is that
reasoning measured, because it had never been separated from the thing it was
being blamed on: every figure in #354, #363, #374 and #383 came from a track
that codes its 768 frames as a single group of pictures, so "an exact seek is
slow" and "an exact seek on a single-group-of-pictures track is slow" had the
same evidence behind them.

`benches/exact_seek.rs` separates them. It measures a cold
`ExactFrameReader::get` — a fresh reader per iteration, nothing cached, nothing
warm, construction off the clock — to the frame four fifths of the way along the
track, across two axes: the backend (`HardwarePreference::Avoid` for the crate's
own software decoder, `Prefer` for the host's fixed-function one) and the
track's random-access cadence. The cadence axis is the pair of fixtures in
`tests/fixtures/codec/`: the same 768 frames of the bundled sample re-encoded at
512x288 with the same encoder, preset and quality, differing only in `keyint`,
so one carries one random-access point and the other twenty-four and nothing
else about them differs.

```sh
cargo bench --bench exact_seek                       # the hardware arms
ZVIDLIB_BENCH_LARGE=1 cargo bench --bench exact_seek # and the software ones
```

The software arms are opt-in for the usual reason: one cold `raps=1` software
seek decodes 613 pictures.

### The measurement

Apple Silicon (M1, 8 cores), `--release`, VideoToolbox as the hardware backend,
seeking to frame 614 of 768. Criterion's median of ten samples.

| track | hardware | software |
| --- | --- | --- |
| `raps=1` (one random-access point) | **119.29 ms** | **1.6556 s** |
| `raps=24` (one every 32 frames) | **18.77 ms** | **31.17 ms** |
| a `PreviewIndex::nearest` lookup over the same track | **1.109 µs** | — |

### What it says

**The cadence is the whole cost, and the two cases do not behave alike.** On the
same content at the same size and quality, one random-access point costs
**6.4x** what twenty-four cost on the hardware backend and **53x** on the
software one. Nothing else about the two tracks differs, so nothing else can be
charged for it.

**A track with several random-access points is already inside the budget, on
both backends.** 18.77 ms and 31.17 ms are both under #374's 50 ms, from cold,
with no preview tier, no proxy and no change to the reader. That is the figure
that **rejects decoding the group of pictures in parallel across several
sessions**: it only ever helps a track with more than one random-access point,
and on such a track the exact seek already answers in 19-31 ms. On the track
that actually has the problem there is nothing to parallelise — one entry point
admits one walk — so the direction is fastest exactly where it is not needed and
inapplicable where it is.

**A track with one random-access point is not reachable by any arrangement of
one decoder.** 119.29 ms is at 512x288; the same seek on the bundled 1080p
sample is the 1.09 s of hardware decode #383 instrumented. Both are the hardware
running at its own throughput with a full queue, so the gap to 50 ms is not a
decoder-configuration problem and no amount of pipelining closes it.

**What is chosen is the preview tier**, now `zvidlib::PreviewIndex` rather than a
copy inside `examples/native_gl`. A lookup is **1.109 µs**, five orders of
magnitude under the 119.29 ms exact seek it stands in for and the only arm in
the table that is under 50 ms for a single-random-access-point track on either
backend. It answers a different question — "what is at this point of the movie",
not "which frame is the pointer on" — which is precisely why it can: the picture
was decoded already. It costs one forward decode pass over the track, on a
thread and a decoder of its own - **2.74 s** to cover all 768 frames of the
bundled 1080p sample through VideoToolbox, measured by
`tests/preview_index.rs::a_preview_answers_any_position_without_decoding`, which
moved out of the example with the tier - and a bounded amount of memory
(`PreviewOptions::budget_bytes`, 64 MB by default, which the stride follows from
so a long track keeps previews further apart rather than more of them).

**A proxy or re-indexed representation is rejected**, and the figure that rejects
it is the 18.77 ms above. A proxy is a track with more random-access points; the
best it can buy is the `raps=24` row, and that row is four orders of magnitude
slower than the preview tier a scrub actually needs. So it pays a full transcode
of the source and a second copy of the media on disk to land in the same range
the exact walk already reaches, on the cases where that range is reachable at
all. It stays rejected until a caller appears that needs *exact* frames at
arbitrary positions repeatedly, which is an editor's requirement rather than a
scrub's, and the preview tier is what a scrub was asking for.

**Nothing about the decoded frames changes.** The preview tier is additive and
opt-in, the reader is untouched, and the 768-frame fixture digests in
`tests/codec_conformance.rs` and `tests/native_hevc_hardware.rs` still hold.

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
instead. Three is a floor rather than a target: the aarch64 table below was
drawn with six, because the host measuring it was running other work and more
rounds is the only lever this recipe has against that.

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
when the table was drawn. When #389 added it, that was three rows of the Apple
M1 table — `hevc_color_convert`, `av1_encode_stage_tile` and
`hevc_encode_640x352_reconstruct` — and nothing on the x86_64 one. #368 re-drew
the Apple M1 table in answer, so both tables are clean today: each is stamped at
a commit carrying the same eleven sites the crate has now, and the report flags
nothing.

Reporting nothing is not the same as being current, and the x86_64 table is
where that showed. It was clean by this check for its whole life — no site
landed under it — while four repairs to kernels behind sites that already
existed moved six of its rows anyway, which is what #393 re-drew it for. This
check answers "does a row name a site that did not exist yet"; it cannot answer
"did the kernel behind an existing site change", because that needs the commit
built rather than read.

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
answered from a commit that is not built. "Is it the *same* vector kernel" is a
third, and the one the x86_64 re-draw above turned on.

### One group name, two targets, and the row that moved for nothing

A third way a committed row stops describing what it names, and the one that is
hardest to see from the table: two bench targets registering the same criterion
group name. Criterion keys a group by its name alone. The name carries no
namespace for the target that registered it and every target in this crate
writes the same `target/criterion/` tree, so two targets both calling a group
`av1_deblock` write the same `target/criterion/av1_deblock/<isa>` directory.
Both groups still run, both print their timings and both pass their
bit-exactness guard; the target that runs second simply overwrites the first,
and `criterion_baseline.py collect` can only ever see one of them.

`benches/codec.rs` and `benches/av1_decode.rs` both claimed `av1_deblock` until
issue #414, and they do not measure the same thing: `codec.rs` filters a
synthetic 1080p luma plane at level 24, `av1_decode.rs` a structured plane at
level 32. **Their vector arms agree to 0.1% and their `scalar` arms are 27%
apart** — the vector kernels do fixed masked work per lane, while the scalar
reference branches per position on the §7.14.6.1 filter mask, so it is the only
arm the content reaches. Which target ran second therefore moved the `scalar`
column of that row by a quarter and left the two vector columns where they were.

That is what #414 reports. The two x86_64 draws ran the targets in opposite
orders: the #350 workflow ran a plain `cargo bench`, which walks targets
alphabetically and so runs `codec` after `av1_decode`, while the #393 workflow
followed the recipe above, which names `codec` first. Read out of #350's own six
rounds, on the three that landed on the AMD EPYC 7763, the two groups are:

| Target's group | `scalar` | `sse4.1` | `avx2` |
| --- | ---: | ---: | ---: |
| `codec.rs`, collected | 21.506 ms | 3.974 ms | 3.424 ms |
| `av1_decode.rs`, overwritten | 27.290 ms | 3.974 ms | 3.403 ms |

so the 21.506 ms the superseded table carried and the 26.956 ms in the x86_64
table below are two different benchmarks, not one benchmark two months apart.

Nothing in the range between the stamps moves the arm. Bisecting it needs a step
comparable to every other step, which the recipe above cannot give — a round is
attributable only to the CPU model it landed on, `ubuntu-latest` is a pool of
models, and this arm reads anywhere between 21.4 ms and 29.9 ms across the pool
at one commit. So each of nine steps over the 32 commits between `b233f0a74f88`
and `605f9a43a24c` measured *both* ends on its own runner, in two worktrees with
their own target directories and three interleaved passes each, and reported the
ratio: **0.977x to 1.000x, every step**, the later commit never slower. A
paired ratio is immune to which model the step landed on, which is what makes
nine of them comparable without dispatching a lottery per step.

**Neither committed table has a row for both groups.** `codec.rs`'s group is now
`av1_deblock_luma`, pairing with the `av1_deblock_chroma` it is the luma half
of, and `av1_decode.rs` keeps `av1_deblock` as the narrow-filter member of its
deblocking trio, the name the recipe's own target order already resolves to.
Both tables below were drawn in that order and so collected `av1_decode.rs`'s
side: each carries an `av1_deblock` row as measured and neither has an
`av1_deblock_luma` row. Each gains it at its next draw. No ratio in either table
is wrong now that it names the group it was measured from.

`no_two_bench_targets_register_the_same_group_name` in
`tests/bench_group_names_are_unique.rs` keeps it fixed, in the `Rust checks`
job. It reads the group names out of each `[[bench]]` target's crate root and
fails on one that two targets claim, which is a check that costs a file read
rather than a draw — and unlike the staleness report it fails rather than
reports, because a collision is a naming mistake in the tree in front of it, not
a measurement to redraw.

### Apple M1 (aarch64)

Measured on **Apple M1 (macOS 26, aarch64)**, at `f3e7674fc5be`, with
`ZVIDLIB_BENCH_LARGE=1` — so this table carries the `_1080p` rows too — and the
elementwise minimum taken across **six** rounds rather than three. Six because
the measuring host was not quiet: it ran other work throughout, at load averages
between 3 and 15 on eight cores, and more rounds is the only lever the recipe
gives against that. Every round logged `scalar` and `neon` and nothing else.

This replaces the table drawn at `b6655bad215f`, whose stamp predates three of
the dispatch sites it recorded. `criterion_baseline.py staleness` names them,
and one re-draw clears all three at once:

| Row | Site that landed after `b6655bad215f` | At the old stamp | Here |
| --- | --- | ---: | ---: |
| `av1_encode_stage_tile` | `av1_coeff_ctx` (#253) | 0.96x | 1.06x |
| `hevc_color_convert` | `hevc_color_convert` (#222) | 1.27x | 3.62x |
| `hevc_encode_640x352_reconstruct` | `hevc_recon` (#208) | 0.98x | 1.97x |

All three were sub-parity or near-parity rows measured before the kernel they
name existed, which is the only thing they had in common; none of them is a
near-parity row now. The table has also grown by twenty-six rows that had no
`neon` figure anywhere: every `_1080p` row, both `av1_encode_stage_coeff_ctx`
rows, both `av1_encode_stage_iwht` rows, `hevc_decode`, `hevc_decode_to_picture`
and the `hevc_encode_1920x1088` family.

The `av1_deblock` row here is `benches/av1_decode.rs`'s group, which this draw
ran second and so collected; there is no `av1_deblock_luma` row, because
`benches/codec.rs`'s group was overwritten in every round. That is the opposite
side of the collision the x86_64 table below collected. See [One group name, two
targets](#one-group-name-two-targets-and-the-row-that-moved-for-nothing) above.

| Group | `scalar` | `neon` | Best |
| --- | ---: | ---: | ---: |
| `av1_cdef` | 35.043 ms | 25.967 ms (1.35x) | 1.35x `neon` |
| `av1_deblock` | 23.124 ms | 3.364 ms (6.87x) | 6.87x `neon` |
| `av1_deblock_boundary` | 276.117 µs | 57.595 µs (4.79x) | 4.79x `neon` |
| `av1_deblock_chroma` | 11.729 ms | 5.131 ms (2.29x) | 2.29x `neon` |
| `av1_deblock_wide` | 58.572 ms | 27.485 ms (2.13x) | 2.13x `neon` |
| `av1_decode_frame` | 76.003 ms | 75.990 ms (1.00x) | 1.00x `neon` |
| `av1_encode_frame_q0` | 17.031 ms | 16.317 ms (1.04x) | 1.04x `neon` |
| `av1_encode_frame_q0_1080p` | 177.864 ms | 170.023 ms (1.05x) | 1.05x `neon` |
| `av1_encode_frame_q160` | 266.334 ms | 172.617 ms (1.54x) | 1.54x `neon` |
| `av1_encode_frame_q160_1080p` | 2.114 s | 1.359 s (1.56x) | 1.56x `neon` |
| `av1_encode_frame_q32` | 243.043 ms | 150.656 ms (1.61x) | 1.61x `neon` |
| `av1_encode_frame_q32_1080p` | 2.402 s | 1.358 s (1.77x) | 1.77x `neon` |
| `av1_encode_stage_bitstream` | 13.678 µs | 14.249 µs (0.96x) | 0.96x `neon` |
| `av1_encode_stage_bitstream_1080p` | 152.156 µs | 156.234 µs (0.97x) | 0.97x `neon` |
| `av1_encode_stage_coeff_ctx` | 1.917 ms | 897.883 µs (2.14x) | 2.14x `neon` |
| `av1_encode_stage_coeff_ctx_1080p` | 17.125 ms | 8.320 ms (2.06x) | 2.06x `neon` |
| `av1_encode_stage_iwht` | 332.657 µs | 308.618 µs (1.08x) | 1.08x `neon` |
| `av1_encode_stage_iwht_1080p` | 3.234 ms | 2.898 ms (1.12x) | 1.12x `neon` |
| `av1_encode_stage_symbol` | 912.771 µs | 923.899 µs (0.99x) | 0.99x `neon` |
| `av1_encode_stage_symbol_1080p` | 8.597 ms | 8.604 ms (1.00x) | 1.00x `neon` |
| `av1_encode_stage_tile` | 16.784 ms | 15.824 ms (1.06x) | 1.06x `neon` |
| `av1_encode_stage_tile_1080p` | 162.570 ms | 142.799 ms (1.14x) | 1.14x `neon` |
| `av1_encode_stage_wht` | 312.714 µs | 304.506 µs (1.03x) | 1.03x `neon` |
| `av1_encode_stage_wht_1080p` | 2.962 ms | 2.864 ms (1.03x) | 1.03x `neon` |
| `av1_entropy_symbol` | 3.075 ms | 3.141 ms (0.98x) | 0.98x `neon` |
| `av1_forward_adst_8x8` | 29.647 ms | 7.272 ms (4.08x) | 4.08x `neon` |
| `av1_forward_dct_16x16` | 37.029 ms | 10.919 ms (3.39x) | 3.39x `neon` |
| `av1_forward_dct_32x32` | 53.549 ms | 62.642 ms (0.85x) | 0.85x `neon` (superseded; see [below](#reading-the-sub-parity-rows)) |
| `av1_forward_dct_4x4` | 40.458 ms | 6.991 ms (5.79x) | 5.79x `neon` |
| `av1_forward_dct_8x8` | 29.153 ms | 7.741 ms (3.77x) | 3.77x `neon` |
| `av1_forward_flipadst_16x16` | 34.389 ms | 11.904 ms (2.89x) | 2.89x `neon` |
| `av1_intra_directional` | 25.517 ms | 25.300 ms (1.01x) | 1.01x `neon` |
| `av1_intra_paeth` | 2.937 ms | 2.944 ms (1.00x) | 1.00x `neon` |
| `av1_intra_smooth` | 2.949 ms | 2.928 ms (1.01x) | 1.01x `neon` |
| `av1_inverse_adst_8x8` | 35.012 ms | 16.710 ms (2.10x) | 2.10x `neon` |
| `av1_inverse_dct_16x16` | 22.046 ms | 11.769 ms (1.87x) | 1.87x `neon` |
| `av1_inverse_dct_32x32` | 17.420 ms | 10.280 ms (1.69x) | 1.69x `neon` |
| `av1_inverse_dct_4x4` | 66.590 ms | 24.599 ms (2.71x) | 2.71x `neon` |
| `av1_inverse_dct_64x64` | 18.831 ms | 11.049 ms (1.70x) | 1.70x `neon` |
| `av1_inverse_dct_8x8` | 36.558 ms | 15.252 ms (2.40x) | 2.40x `neon` |
| `av1_inverse_flipadst_16x16` | 22.783 ms | 13.007 ms (1.75x) | 1.75x `neon` |
| `av1_mc_blend_mask` | 22.952 ms | 10.769 ms (2.13x) | 2.13x `neon` |
| `av1_mc_compound_average` | 22.057 ms | 10.513 ms (2.10x) | 2.10x `neon` |
| `av1_mc_single` | 13.572 ms | 5.392 ms (2.52x) | 2.52x `neon` |
| `av1_motion_compensation` | 13.476 ms | 5.216 ms (2.58x) | 2.58x `neon` |
| `av1_self_guided` | 7.165 ms | 2.823 ms (2.54x) | 2.54x `neon` |
| `av1_wiener` | 7.818 ms | 6.253 ms (1.25x) | 1.25x `neon` |
| `hevc_cabac` | 2.303 ms | 2.361 ms (0.98x) | 0.98x `neon` |
| `hevc_color_convert` | 10.315 ms | 2.851 ms (3.62x) | 3.62x `neon` |
| `hevc_deblock` | 15.417 ms | 13.604 ms (1.13x) | 1.13x `neon` |
| `hevc_decode` | 393.298 ms | 351.123 ms (1.12x) | 1.12x `neon` |
| `hevc_decode_to_picture` | 360.844 ms | 332.305 ms (1.09x) | 1.09x `neon` |
| `hevc_encode_1920x1088` | 871.129 ms | 407.339 ms (2.14x) | 2.14x `neon` |
| `hevc_encode_1920x1088_fwd_transform_quant` | 129.161 ms | 68.830 ms (1.88x) | 1.88x `neon` |
| `hevc_encode_1920x1088_pcm_write` | 4.779 ms | 4.689 ms (1.02x) | 1.02x `neon` |
| `hevc_encode_1920x1088_rdo_inter` | 695.594 ms | 287.745 ms (2.42x) | 2.42x `neon` |
| `hevc_encode_1920x1088_rdo_intra` | 31.857 ms | 15.208 ms (2.09x) | 2.09x `neon` |
| `hevc_encode_1920x1088_reconstruct` | 82.809 ms | 42.166 ms (1.96x) | 1.96x `neon` |
| `hevc_encode_1920x1088_reconstruct_quantized` | 195.560 ms | 91.223 ms (2.14x) | 2.14x `neon` |
| `hevc_encode_1920x1088_residual_write` | 2.135 s | 1.585 s (1.35x) | 1.35x `neon` |
| `hevc_encode_1920x1088_rgba_to_yuv420` | 4.305 ms | 1.206 ms (3.57x) | 3.57x `neon` |
| `hevc_encode_640x352` | 75.976 ms | 35.111 ms (2.16x) | 2.16x `neon` |
| `hevc_encode_640x352_fwd_transform_quant` | 13.618 ms | 7.250 ms (1.88x) | 1.88x `neon` |
| `hevc_encode_640x352_pcm_write` | 369.736 µs | 370.587 µs (1.00x) | 1.00x `neon` |
| `hevc_encode_640x352_rdo_inter` | 77.495 ms | 30.734 ms (2.52x) | 2.52x `neon` |
| `hevc_encode_640x352_rdo_intra` | 3.300 ms | 1.567 ms (2.11x) | 2.11x `neon` |
| `hevc_encode_640x352_reconstruct` | 9.086 ms | 4.604 ms (1.97x) | 1.97x `neon` |
| `hevc_encode_640x352_reconstruct_quantized` | 21.297 ms | 9.552 ms (2.23x) | 2.23x `neon` |
| `hevc_encode_640x352_residual_write` | 214.550 ms | 160.240 ms (1.34x) | 1.34x `neon` |
| `hevc_encode_640x352_rgba_to_yuv420` | 469.579 µs | 123.619 µs (3.80x) | 3.80x `neon` |
| `hevc_encode_bitwriter` | 646.662 µs | 667.521 µs (0.97x) | 0.97x `neon` |
| `hevc_encode_cabac` | 1.711 ms | 1.880 ms (0.91x) | 0.91x `neon` |
| `hevc_encode_cabac_bypass` | 1.972 ms | 1.992 ms (0.99x) | 0.99x `neon` |
| `hevc_inter_pred` | 22.855 ms | 20.458 ms (1.12x) | 1.12x `neon` |
| `hevc_intra_pred` | 8.160 ms | 8.428 ms (0.97x) | 0.97x `neon` |
| `hevc_inverse_transform` | 7.750 ms | 6.096 ms (1.27x) | 1.27x `neon` |
| `hevc_sao` | 2.335 ms | 1.583 ms (1.47x) | 1.47x `neon` |

#### Reading the sub-parity rows

An arm below `1.00x` is slower under its vector kernel than under scalar. Ten
rows are, and eight of them are groups whose two arms are the same code:
`av1_encode_stage_bitstream` and its `_1080p` row, `av1_encode_stage_symbol`,
`av1_entropy_symbol`, `hevc_cabac`, `hevc_encode_bitwriter`, `hevc_encode_cabac`
and `hevc_encode_cabac_bypass` have no vector path at all, so their columns
differ only by measurement noise. The widest that noise got is
`hevc_encode_cabac` at **0.91x**, across a six-round minimum on a host running
other work, and that is the scale to read the rest of the near-parity band at: a
9% gap on this table is not a finding.

The two rows that are not same-code are `av1_forward_dct_32x32` at **0.85x** and
`hevc_intra_pred` at **0.97x**. Both have kernels, and both are rows the
previous draw's discussion had already flagged as unsettled: it recorded three
independent measurement sets reading `hevc_intra_pred` at 0.62x, 1.04x and 1.09x
and `av1_forward_dct_32x32` at 0.78x, 0.91x and 0.95x with no code change
between them, and read the walk as the host getting quieter. This draw is a
minimum over six rounds rather than a single set, which is a tighter estimator
than any of those three — and it does not lift either row to parity. A minimum
removes a round that was contended; it cannot remove contention that was present
in every round, and this host had some in all six.

So the walk-to-parity reading survives for `hevc_intra_pred`, which lands inside
the same 9% band the same-code rows define. It did not survive for
`av1_forward_dct_32x32`, whose 15% gap was the largest sub-parity figure on the
table and sat against 1.48x for the same group under `avx2` on x86_64. That was
the one aarch64 row worth acting on rather than re-reading, and #403 acted on
it; nothing else here is below parity for a reason other than having no kernel.

**`av1_forward_dct_32x32` was the kernel, and the kernel is fixed (#403).** The
row was not the host, and the way to tell was to stop comparing the two arms and
compare the `neon` arm against itself. Cost per vector operation is a ratio
internal to one arm, so contention inflates every size of it equally and cancels
out of the comparison. The 4-, 8- and 16-point forward kernels all ran at
roughly the same cost per operation; the 32-point kernel ran about 2.9x slower
per operation, on the same instruction mix. A cliff at one size is not something
a noisy host can produce.

What it was: `av1_simd::transforms::basis_row_rs14` folded its split
accumulator every fourth term as `if index % 4 == 3`, inside the innermost loop
of an `O(N^2)` basis multiply. At 4, 8 and 16 points LLVM fully unrolls that
loop and the condition costs nothing. At 32 points it stops unrolling and the
condition becomes a real branch per term. Writing the fold as the tail of a
`chunks_exact(4)` group is the same arithmetic in the same order — `N` is always
a multiple of four — and removes the branch.

Measured on the same Apple M1, arms interleaved slab-by-slab with an elementwise
minimum over twelve rounds rather than run group-at-a-time, which is what lets a
contended host still give a usable floor:

| Group | `scalar` | `neon` before | `neon` after |
| --- | ---: | ---: | ---: |
| `av1_forward_dct_4x4` | 42.442 ms | 6.604 ms (5.84x) | 6.746 ms (6.29x) |
| `av1_forward_dct_8x8` | 31.830 ms | 7.312 ms (4.26x) | 7.571 ms (4.20x) |
| `av1_forward_dct_16x16` | 39.675 ms | 10.971 ms (3.37x) | 12.026 ms (3.30x) |
| `av1_forward_dct_32x32` | 62.405 ms | 61.982 ms (0.91x) | 28.563 ms (2.18x) |

The three smaller sizes are unchanged, because they were already unrolled; only
32 points moves, and it moves 2.2x. The scalar column is this round's, not the
table's above: the host was busier here, which is why its `scalar` figures read
high and the before-ratio reads 0.91x where the six-round table read 0.85x. That
difference is the whole reason the row survived three earlier readings — the
scalar arm is the one that moves with the host, the `neon` arm read 62 ms under
both a quiet host and a load average of 17, and a minimum can only be pushed up
by contention, never down. Two arms whose floors are 14% apart under the *same*
contention were never going to be explained by that contention.

**`av1_encode_stage_wht` is no longer 2.72x.** The old table put the forward
4x4 WHT at 2.72x `neon`, and `src/av1_simd/mod.rs` cited exactly that figure as
why `fwht4x4` keeps its kernel on aarch64 while returning `None` on x86_64. This
draw reads it at **1.03x** at 320x180 and 1.03x at 1080p, and reads the inverse
direction — `av1_encode_stage_iwht`, which the old table had no row for at all —
at 1.08x and 1.12x against 0.83x/0.89x under `sse4.1`/`avx2`. The kernel is
still being dispatched: `fwht4x4` falls back to scalar only when an input
exceeds `WHT_INPUT_LIMIT` (2^18) and this group's residuals are a `u8` plane
less a predictor of 128, bounded by 127. So the forward direction is in the
parity band on this host too, and only the inverse still clears it. The
dispatch decision the comment defends is unchanged — `neon` keeps both kernels,
because neither is *below* parity the way the x86_64 arms are — but the number
it defends the decision with is this table's, not the old one's.

The rest of the near-parity rows are the story this file tells above: under
`lto = "fat"` with `codegen-units = 1`, LLVM does to the scalar reference roughly
what the hand kernel does, and the two land within noise of each other.

This is the single most useful thing the committed table records. A one-off
measurement of any of these rows would have looked like a broken kernel and sent
someone rewriting code that was fine. It is also why the CI job compares medians
rather than means, sets its threshold at a deliberately loose 15%, and reports
instead of failing: a shared runner is a noisier host than this one, not a
quieter one.

### x86_64 with SSE4.1 and AVX2 (Linux)

Measured on a GitHub `ubuntu-latest` runner rather than on this project's
development machine, because no aarch64 host can produce these columns at all.
The rounds ran with `ZVIDLIB_BENCH_LARGE=1` and the elementwise minimum was
taken across three of them, exactly as the recipe above describes — the same
recipe the Apple M1 table above is drawn with, so the `_1080p` rows are on both.

GitHub's `ubuntu-latest` pool is not uniform, so the CPU model is checked before
a round is used: an elementwise minimum taken across different CPU models is
attributable to no named host, and the whole point of naming one is that the
numbers are not interchangeable. Six rounds were dispatched at once so that
three sharing a model could be selected afterwards. Three landed on the AMD EPYC
7763 64-Core Processor this table names and are the draw; two landed on an Intel
Xeon 6973P-C and one on an AMD EPYC 9V74 80-Core, and were measured and
discarded. Every merged round logged `scalar`, `sse4.1` and `avx2` in its
`# host instruction sets:` line, and all eleven dispatch sites on `avx2` in its
per-site log.

Measured on **AMD EPYC 7763 64-Core Processor (Linux, x86_64)**, at
`d39c8df519d5` — `605f9a43a24c` (the tip of `main` when the draw was dispatched)
plus the temporary six-round workflow that measured it, which touches no crate
code ([run
33638052798](https://github.com/lsegal/zvidlib/actions/runs/33638052798)).

This table replaces the one drawn at `b284c38a6391`, on the same CPU model, and
why it needed replacing is the half of staleness that [Checking a table still
describes the crate](#checking-a-table-still-describes-the-crate) cannot see. No
dispatch site landed under that draw, so `criterion_baseline.py staleness`
reported it clean for its whole life; four repairs nevertheless moved six of its
rows out from under it. #362 and #371 (PR #385) rebuilt the `av1_coeff_ctx`
routing and then its kernel under the two `av1_encode_stage_coeff_ctx` rows, and
#370 and #387 (PR #394) rebuilt `rdcost`'s block routing and then its batched
motion search under the two `hevc_encode_*_rdo_inter` rows and the two
whole-frame `hevc_encode_*` rows above them. Each repair was recorded in a
re-measurement section of its own rather than folded into that table, which is
the right call per repair — a table attributable to no single host is the
failure mode the six-round selection exists to avoid — and the cost of taking it
four times was a reader chasing four sections to know what the crate does. This
draw folds them back in: every row is re-measured on one named host at one
commit, and the four sections below stay as the record of their own repair
rather than as the numbers to quote.

The draw before that one is #261's, at `e115506f8bf6` on an AMD EPYC 9V74. It
predates the codegen repair in #337 (issue #336), so every `av1_*` row in it
timed `av1_simd` kernels whose `#[target_feature]` wrappers had degenerated into
tail calls to baseline-instruction-set copies — each intrinsic an out-of-line
`core_arch` call with its operand spilled to the stack. It recorded
`av1_deblock_wide` at 0.13x under `avx2` and `av1_forward_flipadst_16x16` at
0.20x. Those figures described the compiler's output, not the kernels, and the
kernels they described no longer exist. `e115506f8bf6` is also a checkpoint
commit on the #257 branch whose merge base with `main` is `b9995b1` (#254), so it
does not contain `f695a1a`, the #222 merge, even though #222 landed on `main`
fifty minutes before the checkpoint was written. That is the whole of why
`hevc_color_convert` moved; see [Reading the rows](#reading-the-rows) below.

The `av1_deblock` row here is `benches/av1_decode.rs`'s group, which this draw
ran second and so collected; there is no `av1_deblock_luma` row, because
`benches/codec.rs`'s group of that name was overwritten in every round. That is
the opposite side of the collision from the table this one supersedes, which is
why the row moved by a quarter in its `scalar` column and not at all in its
vector ones. See [One group name, two
targets](#one-group-name-two-targets-and-the-row-that-moved-for-nothing) above.

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `av1_cdef` | 89.305 ms | 40.075 ms (2.23x) | 32.341 ms (2.76x) | 2.76x `avx2` |
| `av1_deblock` | 26.956 ms | 3.934 ms (6.85x) | 3.392 ms (7.95x) | 7.95x `avx2` |
| `av1_deblock_boundary` | 373.108 µs | 74.155 µs (5.03x) | 78.531 µs (4.75x) | 5.03x `sse4.1` |
| `av1_deblock_chroma` | 15.974 ms | 6.083 ms (2.63x) | 6.304 ms (2.53x) | 2.63x `sse4.1` |
| `av1_deblock_wide` | 104.984 ms | 37.076 ms (2.83x) | 33.176 ms (3.16x) | 3.16x `avx2` |
| `av1_decode_frame` | 97.296 ms | 97.660 ms (1.00x) | 98.031 ms (0.99x) | 1.00x `sse4.1` |
| `av1_encode_frame_q0` | 21.887 ms | 18.474 ms (1.18x) | 18.307 ms (1.20x) | 1.20x `avx2` |
| `av1_encode_frame_q0_1080p` | 205.108 ms | 174.690 ms (1.17x) | 173.326 ms (1.18x) | 1.18x `avx2` |
| `av1_encode_frame_q160` | 281.977 ms | 185.540 ms (1.52x) | 181.625 ms (1.55x) | 1.55x `avx2` |
| `av1_encode_frame_q160_1080p` | 2.592 s | 1.713 s (1.51x) | 1.677 s (1.55x) | 1.55x `avx2` |
| `av1_encode_frame_q32` | 318.283 ms | 213.720 ms (1.49x) | 210.589 ms (1.51x) | 1.51x `avx2` |
| `av1_encode_frame_q32_1080p` | 2.906 s | 1.962 s (1.48x) | 1.915 s (1.52x) | 1.52x `avx2` |
| `av1_encode_stage_bitstream` | 13.498 µs | 13.502 µs (1.00x) | 13.534 µs (1.00x) | 1.00x `sse4.1` |
| `av1_encode_stage_bitstream_1080p` | 128.467 µs | 128.489 µs (1.00x) | 128.449 µs (1.00x) | 1.00x `avx2` |
| `av1_encode_stage_coeff_ctx` | 4.328 ms | 1.409 ms (3.07x) | 1.181 ms (3.66x) | 3.66x `avx2` |
| `av1_encode_stage_coeff_ctx_1080p` | 39.611 ms | 12.976 ms (3.05x) | 10.835 ms (3.66x) | 3.66x `avx2` |
| `av1_encode_stage_iwht` | 405.408 µs | 382.267 µs (1.06x) | 382.240 µs (1.06x) | 1.06x `avx2` |
| `av1_encode_stage_iwht_1080p` | 3.730 ms | 3.521 ms (1.06x) | 3.522 ms (1.06x) | 1.06x `sse4.1` |
| `av1_encode_stage_symbol` | 877.902 µs | 875.397 µs (1.00x) | 872.179 µs (1.01x) | 1.01x `avx2` |
| `av1_encode_stage_symbol_1080p` | 8.088 ms | 8.057 ms (1.00x) | 8.078 ms (1.00x) | 1.00x `sse4.1` |
| `av1_encode_stage_tile` | 21.168 ms | 17.844 ms (1.19x) | 17.684 ms (1.20x) | 1.20x `avx2` |
| `av1_encode_stage_tile_1080p` | 196.224 ms | 165.875 ms (1.18x) | 164.113 ms (1.20x) | 1.20x `avx2` |
| `av1_encode_stage_wht` | 421.947 µs | 364.166 µs (1.16x) | 364.188 µs (1.16x) | 1.16x `sse4.1` |
| `av1_encode_stage_wht_1080p` | 3.889 ms | 3.358 ms (1.16x) | 3.358 ms (1.16x) | 1.16x `sse4.1` |
| `av1_entropy_symbol` | 3.718 ms | 3.718 ms (1.00x) | 3.718 ms (1.00x) | 1.00x `sse4.1` |
| `av1_forward_adst_8x8` | 34.034 ms | 10.029 ms (3.39x) | 9.264 ms (3.67x) | 3.67x `avx2` |
| `av1_forward_dct_16x16` | 38.632 ms | 13.485 ms (2.86x) | 12.156 ms (3.18x) | 3.18x `avx2` |
| `av1_forward_dct_32x32` | 56.210 ms | 22.569 ms (2.49x) | 26.037 ms (2.16x) | 2.49x `sse4.1` |
| `av1_forward_dct_4x4` | 45.110 ms | 7.863 ms (5.74x) | 7.786 ms (5.79x) | 5.79x `avx2` |
| `av1_forward_dct_8x8` | 34.279 ms | 9.956 ms (3.44x) | 9.215 ms (3.72x) | 3.72x `avx2` |
| `av1_forward_flipadst_16x16` | 38.602 ms | 13.814 ms (2.79x) | 12.778 ms (3.02x) | 3.02x `avx2` |
| `av1_intra_directional` | 35.547 ms | 35.537 ms (1.00x) | 35.535 ms (1.00x) | 1.00x `avx2` |
| `av1_intra_paeth` | 3.106 ms | 3.124 ms (0.99x) | 2.996 ms (1.04x) | 1.04x `avx2` |
| `av1_intra_smooth` | 3.083 ms | 3.081 ms (1.00x) | 3.083 ms (1.00x) | 1.00x `sse4.1` |
| `av1_inverse_adst_8x8` | 34.730 ms | 20.214 ms (1.72x) | 22.194 ms (1.56x) | 1.72x `sse4.1` |
| `av1_inverse_dct_16x16` | 26.210 ms | 15.829 ms (1.66x) | 15.087 ms (1.74x) | 1.74x `avx2` |
| `av1_inverse_dct_32x32` | 22.309 ms | 14.023 ms (1.59x) | 13.504 ms (1.65x) | 1.65x `avx2` |
| `av1_inverse_dct_4x4` | 55.009 ms | 26.005 ms (2.12x) | 24.906 ms (2.21x) | 2.21x `avx2` |
| `av1_inverse_dct_64x64` | 27.131 ms | 18.567 ms (1.46x) | 17.945 ms (1.51x) | 1.51x `avx2` |
| `av1_inverse_dct_8x8` | 33.498 ms | 18.578 ms (1.80x) | 18.077 ms (1.85x) | 1.85x `avx2` |
| `av1_inverse_flipadst_16x16` | 28.575 ms | 20.646 ms (1.38x) | 19.424 ms (1.47x) | 1.47x `avx2` |
| `av1_mc_blend_mask` | 25.128 ms | 14.172 ms (1.77x) | 10.826 ms (2.32x) | 2.32x `avx2` |
| `av1_mc_compound_average` | 25.060 ms | 15.655 ms (1.60x) | 11.345 ms (2.21x) | 2.21x `avx2` |
| `av1_mc_single` | 13.108 ms | 6.866 ms (1.91x) | 5.219 ms (2.51x) | 2.51x `avx2` |
| `av1_motion_compensation` | 13.233 ms | 6.849 ms (1.93x) | 5.051 ms (2.62x) | 2.62x `avx2` |
| `av1_self_guided` | 10.775 ms | 3.755 ms (2.87x) | 3.105 ms (3.47x) | 3.47x `avx2` |
| `av1_wiener` | 11.651 ms | 8.438 ms (1.38x) | 6.212 ms (1.88x) | 1.88x `avx2` |
| `hevc_cabac` | 2.188 ms | 2.187 ms (1.00x) | 2.187 ms (1.00x) | 1.00x `avx2` |
| `hevc_color_convert` | 11.911 ms | 3.124 ms (3.81x) | 2.479 ms (4.80x) | 4.80x `avx2` |
| `hevc_deblock` | 14.038 ms | 13.437 ms (1.04x) | 13.428 ms (1.05x) | 1.05x `avx2` |
| `hevc_decode` | 613.511 ms | 472.270 ms (1.30x) | 444.598 ms (1.38x) | 1.38x `avx2` |
| `hevc_decode_to_picture` | 538.099 ms | 452.442 ms (1.19x) | 432.184 ms (1.25x) | 1.25x `avx2` |
| `hevc_encode_1920x1088` | 988.549 ms | 653.553 ms (1.51x) | 467.708 ms (2.11x) | 2.11x `avx2` |
| `hevc_encode_1920x1088_fwd_transform_quant` | 147.884 ms | 98.725 ms (1.50x) | 93.934 ms (1.57x) | 1.57x `avx2` |
| `hevc_encode_1920x1088_pcm_write` | 6.256 ms | 6.212 ms (1.01x) | 6.232 ms (1.00x) | 1.01x `sse4.1` |
| `hevc_encode_1920x1088_rdo_inter` | 874.285 ms | 559.596 ms (1.56x) | 373.683 ms (2.34x) | 2.34x `avx2` |
| `hevc_encode_1920x1088_rdo_intra` | 51.282 ms | 34.314 ms (1.49x) | 34.119 ms (1.50x) | 1.50x `avx2` |
| `hevc_encode_1920x1088_reconstruct` | 95.854 ms | 49.062 ms (1.95x) | 45.449 ms (2.11x) | 2.11x `avx2` |
| `hevc_encode_1920x1088_reconstruct_no_band_search` | 83.923 ms | 38.495 ms (2.18x) | 35.973 ms (2.33x) | 2.33x `avx2` |
| `hevc_encode_1920x1088_reconstruct_quantized` | 221.360 ms | 122.562 ms (1.81x) | 115.279 ms (1.92x) | 1.92x `avx2` |
| `hevc_encode_1920x1088_reconstruct_quantized_no_band_search` | 213.497 ms | 113.656 ms (1.88x) | 106.231 ms (2.01x) | 2.01x `avx2` |
| `hevc_encode_1920x1088_residual_write` | 2.301 s | 1.794 s (1.28x) | 1.686 s (1.37x) | 1.37x `avx2` |
| `hevc_encode_1920x1088_rgba_to_yuv420` | 5.946 ms | 1.197 ms (4.97x) | 963.751 µs (6.17x) | 6.17x `avx2` |
| `hevc_encode_640x352` | 103.204 ms | 68.389 ms (1.51x) | 48.870 ms (2.11x) | 2.11x `avx2` |
| `hevc_encode_640x352_fwd_transform_quant` | 16.085 ms | 10.598 ms (1.52x) | 10.206 ms (1.58x) | 1.58x `avx2` |
| `hevc_encode_640x352_pcm_write` | 724.845 µs | 726.480 µs (1.00x) | 728.722 µs (0.99x) | 1.00x `sse4.1` |
| `hevc_encode_640x352_rdo_inter` | 91.391 ms | 58.827 ms (1.55x) | 39.616 ms (2.31x) | 2.31x `avx2` |
| `hevc_encode_640x352_rdo_intra` | 5.517 ms | 3.694 ms (1.49x) | 3.655 ms (1.51x) | 1.51x `avx2` |
| `hevc_encode_640x352_reconstruct` | 10.064 ms | 5.028 ms (2.00x) | 4.760 ms (2.11x) | 2.11x `avx2` |
| `hevc_encode_640x352_reconstruct_no_band_search` | 8.781 ms | 3.813 ms (2.30x) | 3.504 ms (2.51x) | 2.51x `avx2` |
| `hevc_encode_640x352_reconstruct_quantized` | 23.108 ms | 12.602 ms (1.83x) | 11.837 ms (1.95x) | 1.95x `avx2` |
| `hevc_encode_640x352_reconstruct_quantized_no_band_search` | 22.425 ms | 11.479 ms (1.95x) | 10.496 ms (2.14x) | 2.14x `avx2` |
| `hevc_encode_640x352_residual_write` | 246.162 ms | 191.760 ms (1.28x) | 180.551 ms (1.36x) | 1.36x `avx2` |
| `hevc_encode_640x352_rgba_to_yuv420` | 646.307 µs | 134.182 µs (4.82x) | 105.164 µs (6.15x) | 6.15x `avx2` |
| `hevc_encode_bitwriter` | 703.669 µs | 703.365 µs (1.00x) | 703.663 µs (1.00x) | 1.00x `sse4.1` |
| `hevc_encode_cabac` | 1.691 ms | 1.693 ms (1.00x) | 1.699 ms (1.00x) | 1.00x `sse4.1` |
| `hevc_encode_cabac_bypass` | 2.043 ms | 2.035 ms (1.00x) | 2.045 ms (1.00x) | 1.00x `sse4.1` |
| `hevc_inter_pred` | 28.926 ms | 21.796 ms (1.33x) | 19.668 ms (1.47x) | 1.47x `avx2` |
| `hevc_intra_pred` | 8.402 ms | 8.448 ms (0.99x) | 8.098 ms (1.04x) | 1.04x `avx2` |
| `hevc_inverse_transform` | 9.147 ms | 7.007 ms (1.31x) | 6.600 ms (1.39x) | 1.39x `avx2` |
| `hevc_sao` | 3.129 ms | 1.662 ms (1.88x) | 1.610 ms (1.94x) | 1.94x `avx2` |

#### Reading the rows

Not one row's `Best` arm is below parity, and this is the first x86_64 draw of
which that is true without a footnote: the two `av1_encode_stage_iwht` rows the
previous table marked † at 0.83x and 0.89x read 1.06x and 1.06x here, for
exactly the reason that footnote gave. The lowest cells anywhere in the table
are four readings of 0.99x, and two of them belong to groups whose arms are the
same code: `av1_decode_frame` and `hevc_encode_640x352_pcm_write` have no vector
kernel, so their columns differ only by measurement noise, exactly as the
aarch64 discussion above describes. The same holds for
`av1_encode_stage_bitstream`, `av1_encode_stage_symbol`, `av1_entropy_symbol`,
`av1_intra_directional`, `av1_intra_smooth`, `hevc_cabac`,
`hevc_encode_bitwriter`, `hevc_encode_cabac`, `hevc_encode_cabac_bypass` and the
other `pcm_write` row, which land on `1.00x` from the same cause. The other two
0.99x cells are `av1_intra_paeth`'s and `hevc_intra_pred`'s `sse4.1` arms, which
do have kernels; both groups read 1.04x under `avx2` on the same row, and the
aarch64 table records `av1_intra_paeth` walking from 0.78x to 0.98x across three
draws with no code change, so this is the near-parity band that discussion is
about rather than a kernel to act on.

Three rows read at parity for a reason worth stating rather than as noise, and
the table now shows the reason directly:

- `av1_intra_smooth` is `1.00x` under both vector arms because #337 removed the
  placeholder `smooth_row_{sse41,avx2}` arms. The §7.11.2.6 smooth predictor has
  no vector kernel; those arms only forwarded to `smooth_row_scalar`, and on
  x86_64 a `#[target_feature]` wrapper cannot be inlined into a row loop that
  does not carry the feature, so the forwarding cost a call the aarch64 build
  never paid. That is what the old table's 0.48x was. All three arms now call
  the reference directly, and the row is flat until a real kernel earns the arms
  back.
- `av1_encode_stage_wht` at 1.16x is not a vector win either. #337 routes
  `av1_simd::fwht4x4` to `None` on x86_64 (see #342), because the 4x4 WHT is
  fourteen SSE2-baseline adds, subtracts and shifts that LLVM already
  auto-vectorizes out of `av1_encoder::wht`, against three `transpose4`s of
  shuffle micro-operations the hand kernel adds on top. So all three arms
  execute the same scalar transform, and the ratio is only the input-limit scan
  that the x86_64 early return skips before the fallback — a few percent of a
  very small kernel, not a kernel difference. `neon` keeps the kernel, but not
  the 2.72x this bullet used to cite for it: the re-drawn aarch64 table reads
  the group at 1.03x, in the parity band rather than above it. What still
  separates the two hosts is the *direction* of the gap — aarch64 is at parity
  where x86_64 was under it — and that is what the dispatch turns on.
- `av1_encode_stage_iwht` at 1.06x is now the same story, and this draw is what
  settles it. #342 measured the inverse 4x4 WHT at 0.83x and 0.89x under a hand
  kernel whose two `transpose4`s were sixteen shuffle micro-operations
  contending for one or two ports, against a scalar loop with none, and routed
  `av1_simd::iwht4x4` to `None` on x86_64 in answer. That routing was never
  measured on this table: the two rows carried the pre-change kernel behind a †,
  and the footnote predicted a re-take would read them the way
  `av1_encode_stage_wht` reads. It does. Both rows are above parity rather than
  under it, and the prediction is checkable in the table rather than only in
  prose — the `sse4.1` and `avx2` columns of `av1_encode_stage_wht` (364.166 µs
  against 364.188 µs) and of `av1_encode_stage_iwht` (382.267 µs against
  382.240 µs) agree to four significant figures, across all three rounds, which
  is what "all three arms run the same scalar transform" looks like when the
  arms are genuinely the same code.

**Six rows changed their `Best` arm because four repairs landed under them.**
This is what the draw was for, and each is now the table's own number rather
than a cross-reference:

- `av1_encode_stage_coeff_ctx` reads `avx2` for the first time — 1.181 ms
  against 1.409 ms, 3.66x against 3.07x, and the `_1080p` row 10.835 ms against
  12.976 ms at the same 3.66x, since the group derives contexts for 4x4 blocks
  and nothing else. The previous table had this pair at 1.415 ms `sse4.1`
  against 1.725 ms `avx2`, with `Best` reading 3.04x `sse4.1`. #371's
  `coeff::block_contexts_row_pairs` is what moved it, and the 24% gap that
  change measured on an AMD EPYC 9V74 reproduces at 19% here on the model this
  table names.
- The `hevc_encode_*_rdo_inter` pair reads `avx2` too, and by the largest margin
  in this family anywhere: 39.616 ms against 58.827 ms and 373.683 ms against
  559.596 ms, so 2.31x and 2.34x where the previous table read 1.64x and 1.63x
  under `sse4.1`. That is #387's batched `rdcost::sad_batch`, reproducing to
  within 1% of the single round that measured it (39.723 ms and 373.63 ms
  there).
- The two whole-frame `hevc_encode_*` rows move with them, because the mode
  search is most of what they do. `hevc_encode_640x352` reads 48.870 ms under
  `avx2` against 68.389 ms under `sse4.1` and `hevc_encode_1920x1088`
  467.708 ms against 653.553 ms — both 2.11x, where the previous table had them
  at 1.58x and 1.57x with `Best` reading `sse4.1`. An x86_64 user encoding HEVC
  on an AVX2 host gets about 28% of a whole encode back relative to the arm that
  table would have sent them to.

**Three more rows moved because the thing they measure was rebuilt.** `hevc_sao`
reads 3.129 ms `scalar` against the previous table's 32.116 ms. That tenfold
drop is not a kernel win: #313 rebuilt the group on the real CTB mix rather than
a synthetic one, so it is a different measurement of the same stage, and its
ratio is 1.94x against 1.75x. The two whole-decode groups carry the same change
— `hevc_decode` at 1.38x against 1.21x and `hevc_decode_to_picture` at 1.25x
against 1.11x, with absolute times down from 700.695 ms to 613.511 ms and from
626.111 ms to 538.099 ms — because #313 is what made §8.7.3 reach its vector
kernels in decode at all.

**Four rows are new.** `hevc_encode_640x352_reconstruct_no_band_search`,
`hevc_encode_1920x1088_reconstruct_no_band_search` and the two
`_reconstruct_quantized_no_band_search` rows are the arms #382 added to measure
what share of a reconstruction the SAO band search is; they appear here because
`bench_across_isas` builds them like any other group. They read 2.51x and 2.33x
against their band-searching counterparts' 2.11x and 2.11x, and 2.14x and 2.01x
against 1.95x and 1.92x — the gap being the band search, which
[#382](#why-the-isolated-ratio-did-not-reach-this-group-issue-382) measured at
23-29% of the group's `avx2` arm.

**One row's `Best` moved the other way, and one moved for a reason this draw
cannot name.**

- `av1_forward_dct_32x32` reads 2.49x `sse4.1` against 2.16x `avx2`, where the
  previous table read 1.32x and 1.48x with `avx2` ahead. Both arms got
  substantially faster — 43.147 ms to 22.569 ms and 38.382 ms to 26.037 ms — so
  this is #405's repair of the accumulator fold, which replaced a branch at 32
  points, landing on both widths and on the 128-bit one harder. The three
  merged rounds read 22.57, 22.60 and 22.66 ms against 26.04, 26.06 and
  26.04 ms, so the 13% gap between the arms is a property of the kernels rather
  than of a round. `av1_inverse_adst_8x8` is the only other row where `sse4.1`
  leads by more than the near-parity band, at 1.72x against 1.56x.
- `av1_deblock` reads 7.95x where the superseded table read 6.28x, and the two
  are not the same benchmark. Its `scalar` arm reads 26.956 ms against
  21.506 ms while `sse4.1` and `avx2` land within 1% of their old values
  (3.934 ms against 3.974 ms, 3.392 ms against 3.424 ms), which is the signature
  of the group-name collision #417 settled rather than of a regression: the two
  targets that both registered `av1_deblock` filter different content, so only
  the scalar arm, which branches per position on the filter mask, separates
  them. This draw ran `av1_decode.rs` second and so collected its side; the
  superseded table collected `benches/codec.rs`'s, now `av1_deblock_luma`. A
  nine-step paired bisect over the range read 0.977x to 1.000x at every step,
  so nothing between the stamps moved the arm. See [One group name, two
  targets](#one-group-name-two-targets-and-the-row-that-moved-for-nothing).

The rows with real vector work are the ones with the largest ratios.
`av1_deblock` leads at 7.95x, followed by `hevc_encode_*_rgba_to_yuv420` at
6.17x and 6.15x, `av1_forward_dct_4x4` at 5.79x, `av1_deblock_boundary` at
5.03x and `hevc_color_convert` at 4.80x. The
AV1 forward transforms sit between 3.0x and 3.7x, `av1_self_guided` at 3.47x and
`av1_cdef` at 2.76x, and the motion-compensation family between 2.2x and 2.6x.

**`hevc_color_convert` reads 4.80x, and the 1.00x it replaced was #222's
absence rather than a measurement fault.** Every other row of the
`e115506f8bf6` draw is attributable to #337, and this one is the move #351
recorded without a cause, because #337 touched only `src/av1_simd` and never
`src/hevc/color_convert.rs`. At `e115506f8bf6` there is no
`src/hevc/color_convert.rs` in the tree at all. The conversion is a per-pixel
scalar double loop inside `picture_to_rgba` in `src/hevc/mod.rs`, with no `simd`
dispatch of any kind, so `scalar`, `sse4.1` and `avx2` ran byte-identical code
and `1.00x / 1.00x` is exactly what they should have read. `benches/hevc_decode.rs`
said as much at that commit: its per-stage table listed the group's `Vectorized`
column as "no, today". The `convert_row_{sse41,avx2}` kernels arrived with #222
(`f695a1a`), which the checkpoint the draw was taken on does not contain — see
the paragraph on `e115506f8bf6`'s merge base above. So the 4.80x is #222's win,
and nothing between the draws changed what the group *measures*: what changed is
that there is now something to measure.

This also settled the aarch64 side, and #368 has since re-drawn it. That
table's [sub-parity discussion](#reading-the-sub-parity-rows) named
`hevc_color_convert` as a group whose arms are the same code; that was true of
the draw it described and not of the crate. The row now reads 3.62x `neon` at
`f3e7674fc5be`, measured against a `hevc_color_convert` kernel that exists, and
it was not the only row in that position — `av1_encode_stage_tile` and
`hevc_encode_640x352_reconstruct` predated their sites too.

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
per-stage groups](#the-hevc-per-stage-groups), or a committed table's agreement
with kernels that changed *without* adding a site — the six rows above are
exactly that case, and re-drawing on a schedule is the only answer to it.
Quoting an old table's ratio is only safe alongside the commit stamped on it,
which is why the stamps are there.

The whole-frame encoder groups are the practical consequence.
`av1_encode_frame_q32` reads 1.51x and `av1_encode_frame_q160` 1.55x here,
against 0.52x and 0.48x in the `e115506f8bf6` table: an x86_64 user of the AV1
encoder stops paying roughly twice over for having a vector path and starts
getting about 1.5x back for it. The `_1080p` variants agree to two decimal
places, so the ratio is a property of the kernels rather than of the frame size.

Where `sse4.1` still beats `avx2` it is now by the near-parity band or by
`av1_forward_dct_32x32`'s 13%, and no longer on the two families that made the
`Best` column disagree with what a real encode does. Both of those are settled,
and settled the same way: **the wide arm was not reaching its width on the block
shapes these workloads actually use**, and the answer in each case was to give
it width rather than to route around it. The mechanism differs, and the
re-measurement sections below are the record of how each was established.

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
  The table above now carries the result: this group's `avx2` column is
  `block_contexts_row_pairs` measured on the host the table names, at 3.66x
  against `sse4.1`'s 3.07x, so the two sections below are the record of how the
  redirect and the kernel were established rather than the current numbers.
#### The #362 re-measurement

This section is the record of how #362's redirect was established, not the
crate's current numbers: the table above supersedes it, and reads this group at
3.66x `avx2` against 3.07x `sse4.1` because #371 replaced the redirect with a
kernel. What survives here is the evidence that the redirect did what it
claimed, on the round that measured it.

The repair was measured, not asserted, and it was deliberately recorded here
rather than folded into the table. The `workflow_dispatch` round that measured
it landed on an **Intel(R) Xeon(R) 6973P-C (Linux/X64)** — one of the CPU models
the table's draw explicitly measured and discarded for not being the AMD EPYC
7763 the merged rounds shared — and it is one round, not the elementwise minimum
of three. Its absolute times are therefore not comparable with the table's, and
merging two rows of it into a table attributed to a named host would make that
table attributable to no host at all, which is the failure mode the six-round
selection above exists to avoid.

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

The `rdo_inter` pair is the control, and it is unmoved: 1.69x under `sse4.1`
against 1.54x/1.55x under `avx2`, the same shape #351 recorded on a different
host. Nothing in #362 touches `rdcost`, and the second bullet above is why it
would not have helped if it did. That pair has since been repaired twice on its
own account, by #370 and #387, and the table above now reads it at 2.31x and
2.34x under `avx2`; this round predates both.
#### The #371 re-measurement

#362's repair routed around the idle lanes; #371 removes them, and the
acceptance criterion was that the wide arm has to *win* on its own numbers
before the dispatch site takes it back. It does, and by more than the margin
#362 measured against it.

The table above now carries this result directly, on the AMD EPYC 7763 it names
rather than on the EPYC 9V74 measured here: `av1_encode_stage_coeff_ctx` reads
1.181 ms `avx2` against 1.409 ms `sse4.1` there, a 19% gap against the 24% below,
and `Best` reads `avx2` on both rows. This section stays as the record of the
acceptance criterion and the round that met it.

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
Processor (Linux/X64)** — the same host model both the superseded x86_64 table
and the one above were measured on — so its columns can be read against either
directly. It is still one round rather than the elementwise minimum of three,
which is why it is recorded here rather than merged into a table; the controls
below are what carry the attribution. Every figure it compares against is the
`b284c38a6391` draw, which the table above replaces; #387 has since moved these
rows again, and the table's numbers are that later state rather than this one. Measured at `6213a5580b78` with `ZVIDLIB_BENCH_LARGE=1`,
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
to 55.886 ms and 547.481 ms to 530.010 ms against the superseded table, both about
3% faster, while the same rows' `scalar` and `sse4.1` columns land within 1% of
their table values (88.922 against 89.626, 55.341 against 54.769, 843.440
against 851.069, 523.280 against 520.861, all from that draw). `rdo_intra`, which runs the same
`rdcost::satd` through a mode search that never calls `sad`, is unmoved at
1.48x/1.50x against that draw's 1.49x/1.49x — one round of run-to-run noise on
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
reads 65.257 ms under `avx2` against that draw's 67.101 ms and
`hevc_encode_1920x1088` 628.680 ms against 646.477 ms, so an x86_64 user
encoding HEVC on an AVX2 host gets about 2.8% of a whole encode back — the two
arms are now 0.3% and 0.6% apart where that draw had them 4.3% and 4.0% apart.
#387 turned that 2.8% into about 28%, and the table above is where that reads.
`av1_encode_stage_coeff_ctx` reads 1.4647 ms and 1.4745 ms on this host, 0.7%
apart, which is #362's redirect reproducing on the committed table's own
hardware — a state #371 has since separated again, at 1.181 ms against 1.409 ms
in the table above.

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
  number. Widening what the search hands the kernel is the other repair and
  #387 wrote it — it is the one that makes AVX2 *win* here rather than stop
  losing, and it moves the search's candidate ordering with it, so it is an
  optimization rather than a defect fix. Measured under [The #370
  re-measurement](#the-370-re-measurement) and [The #387
  re-measurement](#the-387-re-measurement); the table above now carries the end
  state of both, at 2.31x and 2.34x `avx2` against `sse4.1`'s 1.55x and 1.56x,
  so those two sections are the record of how each step was established rather
  than the current numbers.

#### The #387 re-measurement

#370 routed the narrow blocks around AVX2 and brought the two arms level; #387
gives AVX2 something wide to do and the acceptance criterion was that it has to
*win* on its own numbers. It does, by more than routing ever had to give.

The width was never in the block — `rdo.rs` searches a `CTB` of 16 and its
candidate partitions subdivide that — it is across the *candidates*: the
whole-pel stage scores `(2 * radius + 1)^2` predictions of one source block, and
`_mm256_sad_epu8` reduces per 8-byte lane, so one instruction carries two
16-wide candidates (one per 128-bit lane) or four narrower ones (one per qword).
`rdcost::sad_batch` is that entry point, `rdo::motion_search` gathers candidates
into fixed batches to feed it, and the scan order and `mv_order` tie-break are
untouched, so the winning vector is the one the per-candidate search picked.
This is also the one place `SAD_AVX2_MIN_W` does not apply: a width the
single-block path routes *away* from AVX2 is exactly a width the batched path
routes *to* it, because its vector is full there.

Measured at `c7205cf2a908` — the branch's merge with `main`, so #370's routing is
in the tree — on an **AMD EPYC 7763 64-Core Processor (Linux/X64)**, the same
host model as the committed table and as #370's round, with
`ZVIDLIB_BENCH_LARGE=1`, `# host instruction sets: scalar, sse4.1, avx2` and
`# dispatch site hevc_rdcost: avx2` ([run
33625305783](https://github.com/lsegal/zvidlib/actions/runs/33625305783)). It is
one round rather than the elementwise minimum of three, so it is recorded here
rather than merged into the table; the `rdo_intra` control below is what carries
the attribution.

| Group | `scalar` | `sse4.1` | `avx2` | Best |
| --- | ---: | ---: | ---: | ---: |
| `hevc_encode_640x352_rdo_inter` | 92.881 ms | 58.789 ms (1.58x) | 39.723 ms (2.34x) | 2.34x `avx2` |
| `hevc_encode_1920x1088_rdo_inter` | 878.39 ms | 558.10 ms (1.57x) | 373.63 ms (2.35x) | 2.35x `avx2` |
| `hevc_encode_640x352` | 104.39 ms | 68.400 ms (1.53x) | 48.776 ms (2.14x) | 2.14x `avx2` |
| `hevc_encode_1920x1088` | 1.0028 s | 656.45 ms (1.53x) | 472.23 ms (2.12x) | 2.12x `avx2` |
| `hevc_encode_640x352_rdo_intra` | 5.598 ms | 3.762 ms (1.49x) | 3.753 ms (1.49x) | 1.49x `avx2` |
| `hevc_encode_1920x1088_rdo_intra` | 51.991 ms | 34.943 ms (1.49x) | 34.256 ms (1.52x) | 1.52x `avx2` |

The `Best` column of the `rdo_inter` pair reads `avx2` for the first time. The
two vector arms are 32% and 33% apart with the wide one ahead — 39.723 ms
against 58.789 ms and 373.63 ms against 558.10 ms — where #370's round on this
same host model had them 1.0% and 1.3% apart, and every round before it had
`sse4.1` ahead by 5%. Level was the whole of what routing could buy; the batched
kernel is what buys more than level.

`rdo_intra` is the control that should not move, and does not: it scores intra
predictions through `satd` alone, forms no batch, and reads 1.49x/1.52x here
against #370's round's 1.48x/1.50x on the same host model. The absolute times of
this round run about 5% slower than that one across every column, `scalar`
included — one draw of run-to-run variance on a shared runner — which is why the
claim is the within-round sign rather than the absolute numbers.

The whole-frame groups are the practical consequence, since the mode search is
most of what they do: `hevc_encode_640x352` reads 48.776 ms under `avx2` against
68.400 ms under `sse4.1` and `hevc_encode_1920x1088` 472.23 ms against
656.45 ms, so an x86_64 user encoding HEVC on an AVX2 host gets about 28% of a
whole encode back, where #370's routing recovered 2.8% of it.

The committed table above no longer pre-dates any of this — it is the reason
this issue's re-draw happened — and it is the check on this round as well as its
successor. Its three merged rounds read `hevc_encode_640x352_rdo_inter` at
39.616 ms and `hevc_encode_1920x1088_rdo_inter` at 373.683 ms against the
39.723 ms and 373.63 ms below, within 1% on both, and the whole-frame pair at
48.870 ms and 467.708 ms against 48.776 ms and 472.23 ms. So the single round
recorded here reproduces as an elementwise minimum of three on the same host
model, which is the strongest form the attribution below can take.

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

The three generated `synthetic_*` fixtures share one moving diagonal ramp, and
its luma wraps at `& 0xff`. That gives a block an atypical spread of sample
values, which matters to any measurement whose cost depends on the distribution
of samples rather than their count; `What the synthetic content's value
distribution is, and what it cannot answer` states what it costs and how to
re-take the figure.

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

### What the synthetic content's value distribution is, and what it cannot answer

The encoder groups decode nothing first, so their content is generated rather
than filmed, and it is generated the same way for all of them:
`support::synthetic_yuv420_sequence` builds a moving diagonal ramp plus
low-amplitude noise for the per-stage groups, `support::synthetic_rgba8_sequence`
the same ramp in RGBA8 for the whole-frame groups, and `support::av1_gray8_planes`
borrows the first's luma outright for the AV1 encoder groups above. That keeps
neither prediction nor entropy coding in an unrepresentative best case, which is
what it was chosen for, and it is a fair input for any kernel whose cost scales
with the *number* of samples it touches.

It is not a fair input for a kernel whose cost depends on the *distribution* of
those samples within a block. Both luma generators close with `& 0xff`, so a ramp
that would otherwise leave 8-bit range wraps back to zero, and a block straddling
the wrap spans nearly the whole 8-bit range instead of the narrow one video's
spatial coherence would give it. The chroma — a `% 24` sawtooth around 128 — has
the opposite skew, being *more* concentrated than video everywhere.

`tests/sao_band_occupancy.rs` is the measurement of this and the way to re-take
it:

```sh
cargo test --features native --release --test sao_band_occupancy -- \
  --ignored --nocapture
```

It reports how many of the 32 §8.7.3.2 SAO bands a block occupies — those bands
are 8-bit value bins three places wide, so this is a within-block value
distribution rather than an SAO-specific quantity — both as distinct bands and as
the band range `max - min + 1`:

| content | distinct bands | band range `max - min + 1` |
| --- | --- | --- |
| bundled 1080p sample, luma 16x16 (n=168,840) | mean 6.4, 38.4% <=4 | mean 6.4, 38.4% <=4, 68.6% <=8 |
| synthetic 640x352 luma 16x16 (n=1,760) | mean 10.0 | **mean 15.5, 0% <=8** |
| synthetic 640x352 chroma 8x8 (n=3,520) | mean 3.2 | mean 3.4, 100% <=4 |

Read the two luma rows against each other. Real video occupies four bands or
fewer in 38.4% of its CTBs; the synthetic luma occupies eight or fewer in **none**
of its 1,760. A kernel with a narrow-block fast path would therefore have run
that path on over a third of real CTBs and on no synthetic luma CTB at all, so
`hevc_encode_640x352_reconstruct` reports it as pure overhead whatever it is
worth on video.

#406 is the worked example. The transposed SAO band kernel it wrote costs work
proportional to the band range it visits, and the group its acceptance criterion
named could not have exercised the fast path even had the kernel deserved one.
That kernel was a null result for a separate and more decisive reason — narrowing
the accumulators loses before any vector unit is involved, which the portable
control measured at 0.66-0.81x — so nothing about its conclusion turns on this,
and the full account is under `The register-resident accumulator` below. The
criterion was still stated against content that could not answer it.

Prediction-mode selection, transform-size selection and anything driven by local
variance carry the same exposure. Before writing a kernel whose cost varies with
value distribution, re-take the measurement above and check that the group meant
to judge it carries the distribution the kernel needs; when it does not, judge it
on the bundled sample's decoded luma, which `tests/sao_band_occupancy.rs` reads
alongside the synthetic planes for exactly that comparison.

The generators are deliberately left as they are. Clamping or reflecting the ramp
instead of wrapping it would make the content locally coherent the way video is,
but it would move every committed encoder row in `Committed baselines` below and
require re-drawing both tables by the recipe there — a larger change than the
exposure warrants now that it is written down.

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

#### Why the isolated ratio did not reach this group (issue #382)

The two measurements above disagree, and #340 recorded two candidate reasons
without separating them: either the band search is too small a share of a
reconstruction for any ratio on it to show, or the classification is worth
vectorizing here and only the *call shape* is wrong, a `#[target_feature]`
kernel being uninlinable into a caller that invokes it once per 16-to-64-sample
CTB row. The two have opposite consequences - the first says no kernel at this
site can ever pay, the second says the site needs a different call - so the
question had to be settled by measurement rather than argument.

**The share is a measured number.** `hevc_encode_*_reconstruct_no_band_search`
is the same reconstruction with the band half of the §8.7.3 search skipped, so
the per-CTB decision is the four edge-offset classes against SAO off. Both arms
are built and timed in one criterion process on one host, five interleaved
rounds, per-benchmark minimum; the difference is what the band search costs.
Six `ubuntu-latest` draws plus one `macos-15-intel`, grouped by CPU model as
every x86_64 table here has to be:

| CPU model | draws | `hevc_encode_640x352_reconstruct` | `avx2` share | `scalar` share |
|---|---|---|---|---|
| Intel Xeon Platinum 8573C | 2 | 3.90-4.47 ms | 25.4%, 23.2% | 9.9%, 10.0% |
| Intel Core i7-8700B | 1 | 7.49 ms | 28.8% | 8.5% |
| AMD EPYC 7763 | 1 | 4.72 ms | 26.2% | 12.7% |
| AMD EPYC 9V45 | 1 | 2.69 ms | 28.8% | 9.8% |
| AMD EPYC 9V74 | 2 | 4.77-4.81 ms | 27.5%, 28.9% | 13.9%, 12.8% |

**So the first explanation is refuted.** The band search is roughly a quarter
of the vectorized arm of this group on every model timed, against a
round-to-round spread of 1-6% on the `ubuntu-latest` hosts. A 1.10-1.53x on
work that big is several percent end to end, which this harness resolves. It
did not appear, so the missing win is not a missing denominator.

Read the two share columns together rather than either alone. The band search
is scalar in both, so the `scalar` column's 8-14% is its share of a
reconstruction whose *other* stages are also scalar, and the `avx2` column's
23-29% is its share once prediction, the transform round trip, deblocking and
the SAO filter itself have been vectorized around it. The second number is the
one that matters for a kernel decision, and it is the one that grew: pinning
everything else made the unvectorized band search the largest single item in
the group. The `_quantized` arms read 8.6-11.6% for the same reason in
reverse - the transform round trip they add is a large scalar-and-vector cost
the band search is then a smaller fraction of.

The `macos-15-intel` draw agrees on the share and should be read only for it:
that host's round-to-round spread is 24-51%, against 1-6% on `ubuntu-latest`,
so its minimum is a floor rather than a measurement.

**The call shape is not the reason either.** That left the second explanation,
which is testable directly: give the kernel a call shape whose per-call cost is
amortized over a whole CTB instead of one row. `band_offset_rect` is that
shape - the rows of a CTB are walked *inside* one `#[target_feature]` entry, so
a 16x16 luma CTB pays one non-inlinable call rather than sixteen and an 8x8
chroma CTB one rather than eight, over the same lane-scatter body #340 timed.
Measured the same way the per-row shape was, as a paired branch-against-base
comparison with both trees built and timed on one host and interleaved within a
round, five rounds per draw:

| CPU model | draws | `avx2` | `sse4.1` | `scalar` (control) |
|---|---|---|---|---|
| Intel Xeon Platinum 8573C | 1 | 1.023x | 1.015x | 1.010x |
| AMD EPYC 7763 | 6 | 0.966-1.003x | 0.993-1.007x | 1.001-1.015x |
| AMD EPYC 9V74 | 5 | 0.872-0.891x | 0.974-0.992x | 1.006-1.008x |

**It is worse, not better.** On Zen 5 the once-per-CTB shape reads 0.872-0.891x
across five independent draws where the once-per-row shape read 0.94-0.95x:
removing fifteen of every sixteen calls made the kernel *more* expensive, which
is the opposite of what a call-overhead account predicts and enough on its own
to refute it. On EPYC 7763 it reproduces the per-row figure, sitting 1-4% under
that draw's own control. On the one readable Intel draw it is 1.023x against a
1.010x control - no separation, the same answer the per-row shape's 1.00x gave.

A twelfth draw, on `macos-15-intel` (Intel Core i7-8700B), is **discarded rather
than reported**: its `scalar` control read 1.172x, and a control that moves 17%
where it must read 1.00x says the host was not quiet enough for a several-percent
effect. That is the control doing its job. The same host's spread disqualified it
from the share table above for the same reason.

**So neither candidate cause survives, and the site is not the problem.** What
the two measurements together say is that the isolated harness is not measuring
the encoder's work: `bench_band_offset_row` runs one L1-resident run of up to
1024 samples back to back with `stats` hot and the branch fully predicted, and
the reconstruction loop runs the same classification over CTB windows of a
picture-sized plane, interleaved with prediction, the transform round trip and
two in-loop filters, with `stats` reloaded per CTB and the band histogram
competing for cache with everything else in the stage. The 1.10-1.53x is a real
figure for the loop it was taken in and does not transfer, and a third call
shape is not what would make it transfer.

That left one untested idea, and it was not a call shape but a different
decomposition - deriving the 32-band histogram for a whole CTB in one pass that
keeps its accumulators in registers across rows, rather than 32 memory
accumulators re-read per row. #406 built it. It is a null result, and this
dispatch site is now closed.

#### The register-resident accumulator, and why it is a null result

Three shapes, all asserted bit-exact against `band_offset_row_scalar` under
every `simd::set_override` pin over strided rectangles by
`the_x86_narrow_band_rect_kernels_match_the_row_reference`: a **portable narrow
control** that narrows and splits the accumulators and vectorizes nothing, so a
result can be attributed to the accumulator rather than to the classification;
**`band_offset_rect_avx2_narrow`** and its SSE4.1 twin, which are the same lane
scatter #340 timed but into `i32` sums and `u32` counts split across two partial
sets instead of two `[i64; 32]` arrays; and
**`band_offset_rect_avx2_transposed`**, the register-resident shape proper - a
masked pass per band with the histogram held in `ymm` registers, no memory
histogram and no scatter at all.

**The occupancy the transposed shape depends on was measured first**, because
its cost is proportional to the bands it visits and only the band *range* is
derivable at a price it can pay: a vector min/max is two operations per sample,
while the set of distinct occupied bands is not available at any price the pass
can afford. `tests/sao_band_occupancy.rs` is that measurement.

| content | distinct bands | band range `max - min + 1` |
| --- | --- | --- |
| bundled 1080p sample, luma 16x16 (n=168,840) | mean 6.4, 38.4% <=4 | mean 6.4, 38.4% <=4, 68.6% <=8 |
| synthetic 640x352 luma 16x16 (n=1,760) | mean 10.0 | **mean 15.5, 0% <=8** |
| synthetic 640x352 chroma 8x8 (n=3,520) | mean 3.2 | mean 3.4, 100% <=4 |

Real video is sparse. The synthetic content this group searches is not, because
its luma wraps a gradient at `& 0xff`. **So the one shape with a sparse-CTB
advantage has no sparse luma CTB to take it on in the group that decides** -
worth knowing before writing it, which is why it was measured before writing it.

**Narrowing the accumulators is a loss on its own, which is the opposite of the
premise.** `bench_band_offset_rect` times the two rectangles the site is
actually called with - a 16x16 luma CTB and an 8x8 chroma CTB - including the
per-CTB zeroing and fold a whole-CTB accumulator has to pay and a per-row one
does not. Intel Core i9-10850K, minimum of 15 interleaved rounds, reproducing to
+/-0.03 across independent runs:

| arm | luma 16x16 | chroma 8x8 |
| --- | ---: | ---: |
| rows (reference) | 1.00x | 1.00x |
| lane scatter into `BandStats` (#340's kernel) | 1.07-1.09x | 0.98-1.02x |
| narrow scalar control | **0.80-0.81x** | **0.66-0.70x** |
| avx2 narrow | 0.99-1.03x | 0.77-0.80x |
| avx2 narrow, fold deleted | 1.06-1.11x | 0.94-0.99x |
| avx2 transposed | 0.84-0.86x | 0.61-0.65x |
| avx2 transposed, 4-band range | 1.05-1.13x | 0.62-0.65x |

Read the control row first: narrowing and splitting the accumulators, with no
vector unit involved at all, reads **0.66-0.81x**. The narrow histogram costs
more than the width it saves, and `u16` counts were worse again than the `u32`
this table was taken at. Two `u32`-counted sets are also 512 bytes - exactly
what `BandStats` costs - so the "halves the traffic" half of the idea and the
"splits the dependency chain" half are in direct tension, and only one of them
can hold at a time.

The `fold deleted` row is the ceiling. It prices away the one cost this shape
adds that a per-row `BandStats` does not, so **no rewiring of the caller to
consume the narrow accumulator directly could beat it**. Even there it reads
1.06-1.11x on luma and 0.94-0.99x on chroma against a wide-scatter kernel
already at 1.07-1.09x and 0.98-1.02x in the same rounds: not distinguishable
from the kernel #340 measured and #382 showed does not transfer.

**And it does not separate in the encoder either**, which is the criterion the
last two attempts failed. Paired branch-against-base, both trees built and timed
on one host and interleaved within a round, five rounds per draw, with
`band_offset_rect` dispatching to the narrow kernels in the branch and to the
row reference in the base. The `scalar` arm is the control: it resolves to the
same row reference in both trees, so it has to read 1.00x.

Eight `ubuntu-latest` legs were dispatched at once so that legs sharing a CPU
model could be grouped afterwards, plus one local Intel desktop host:

| CPU model | draws | `avx2` | `sse4.1` | `scalar` (control) |
| --- | ---: | ---: | ---: | ---: |
| Intel Xeon Platinum 8573C | 2 | 1.006-1.012x | 1.000-1.006x | 1.000-1.001x |
| AMD EPYC 7763 64-Core | 2 | 0.991-1.005x | 0.977-0.998x | 0.999-1.001x |
| AMD EPYC 9V74 80-Core (Zen 5) | 3 | **0.959-0.965x** | 0.967-0.980x | 1.002-1.005x |
| Intel Core i9-10850K (local) | 1 | 0.978x | 1.003x | 1.000x |

**Nothing separates, and on Zen 5 it is a reproducible regression.** On both
Intel parts and on EPYC 7763 every arm sits inside a band its own control is
already inside - 1.006-1.012x against a 1.000-1.001x control is not a result.
On EPYC 9V74 the `avx2` arm reads 0.959-0.965x across three independent draws,
every round signed the same way, against controls of 1.002-1.005x: a 3.5-4%
regression, well outside what the control moves by. That is the same signature
#340's per-row shape (0.94-0.95x on Zen 5) and #382's per-CTB shape
(0.872-0.891x on Zen 5) both produced, and the third time this dispatch site has
answered a differently-shaped kernel the same way.

A ninth leg, on an Intel Xeon 6973P-C, is **discarded rather than reported**:
its `scalar` control read 0.975x, and a control that moves 2.5% where it must
read 1.00x cannot resolve a several-percent effect. That is the control doing
its job. An earlier six-leg draw agreed in range - `avx2` 0.975-1.012x,
`sse4.1` 0.971-1.006x, controls 0.999-1.010x - but is not reported by model,
because that draw wrote the host's model only to the run's step summary and not
to the artifacts the ratios are recomputed from. It is the reason the workflow
now records the model in an uploaded file.

**Do not spend a fourth attempt on this dispatch site.** Three shapes across
three issues now agree, and none of the three guessed causes survived: not the
denominator (#382 measured the band search at 23-29% of this group's `avx2`
arm), not the call shape (#382 measured once-per-CTB as worse than
once-per-row), and not the accumulator width (#406 measured narrowing as a loss
before any vector unit is involved). What is left is that a 32-way scatter over
a CTB is already close to what this classification costs, and the isolated
harness's 1.10-1.53x is a figure for a loop the encoder does not run.
`band_offset_rect` is kept as the site's entry point so the site keeps a named
dispatch point with a bit-exactness harness pointed at it; all five x86 rect
kernels stay `#[cfg(test)]` alongside the six once-per-row candidates, asserted
bit-exact, as the apparatus these figures were taken with.

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
