# `zvidlib-av1-decoder` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`big_buck_bunny_av1_yuv420.sha256` holds one canonical `FrameDigest` per
presentation frame (0-767) of the bundled `examples/media/BigBuckBunny.av1.mp4`
sample (SVT-AV1, 8-bit 4:2:0 Main, 960x540, limited range) as decoded by
FFmpeg's libdav1d wrapper, fingerprinted as `Yuv420p8` frames with limited
color range. They were generated offline with

```sh
ffmpeg -c:v libdav1d -i ../../../../examples/media/BigBuckBunny.av1.mp4 \
  -fps_mode passthrough -f rawvideo -pix_fmt yuv420p ref.yuv
```

and hashed as `FrameDigest::from_frame` does: the big-endian width and height,
the `Yuv420p8` tag (4), the limited-range tag (1), the plane count (3), then
each frame's Y, U and V samples. `crates/zvidlib-av1-decoder/src/av1_dec/tests.rs` checks the crate's AV1
decoder against every one of them. `big_buck_bunny_av1_rgba.sha256` carries the
`Rgba8` digests of the same libdav1d frames after this crate's BT.601
`convert_to_rgba8` (`crates/zvidlib-av1/src/av1_filters.rs`), the conversion
`native_av1_video_decoder_factory` applies to BT.601 (`matrix_coefficients` 6)
streams; `tests/codec_conformance.rs` and the browser fallback test in
`src/web_decoder.rs` compare the factory's output with them.

`av1_main10_64x64.mp4` is a single 64x64 10-bit 4:2:0 AV1 Main frame, which
the software AV1 decoder refuses because it decodes 8-bit streams only. It is
the browser fallback's example of a track neither decoder can take, generated
offline with

```sh
ffmpeg -f lavfi -i testsrc=size=64x64:rate=30 -frames:v 1 -c:v libaom-av1 \
  -cpu-used 8 -crf 40 -pix_fmt yuv420p10le -movflags +faststart av1_main10_64x64.mp4
```

As above, FFmpeg is only the offline fixture generator and is not a build,
test, or runtime dependency.

The `av1_*.mp4` fixtures below each hold six frames of 8-bit 4:2:0 AV1 Main,
coded to reach decoder tools the bundled `BigBuckBunny.av1.mp4` sample never
does (issue #519). Together they are about 58 KB.

| Fixture | Tools it exists to reach |
| --- | --- |
| `av1_palette_intrabc.mp4` | palette mode and intra block copy |
| `av1_film_grain.mp4` | film grain synthesis |
| `av1_superres.mp4` | super-resolution upscaling |
| `av1_qmatrix.mp4` | quantizer matrices (`using_qmatrix`) |
| `av1_tiles.mp4` | 2x2 tiles in two tile groups, with `context_update_tile_id` other than 0 |
| `av1_sb128.mp4` | 128x128 superblocks, at an odd frame size |
| `av1_segmentation_deltas.mp4` | segmentation, delta q and delta loop filter |
| `av1_compound_filters.mp4` | filter intra, wedge, difference-weighted and distance-weighted compound, dual filter |

They were generated offline with FFmpeg 6.0's libaom (3.6.0) and SVT-AV1
(1.4.1) wrappers from the bundled HEVC sample and FFmpeg's `testsrc`:

```sh
B=../../../../examples/media/BigBuckBunny.mp4
ffmpeg -ss 10 -i $B -an -frames:v 6 -vf scale=256:144 -pix_fmt yuv420p bbb.y4m
ffmpeg -ss 20 -i $B -an -frames:v 6 -vf scale=250:142 -pix_fmt yuv420p bbb_odd.y4m
ffmpeg -f lavfi -i testsrc=size=256x144:rate=30 -frames:v 6 -pix_fmt yuv420p screen.y4m
A="-c:v libaom-av1 -cpu-used 1 -movflags +faststart"
ffmpeg -i screen.y4m $A -crf 30 -enable-palette 1 -enable-intrabc 1 \
  -aom-params tune-content=screen av1_palette_intrabc.mp4
ffmpeg -i bbb.y4m $A -crf 40 -aom-params film-grain-test=1 av1_film_grain.mp4
ffmpeg -i bbb.y4m $A -crf 40 -aom-params enable-qm=1:qm-min=0:qm-max=15 av1_qmatrix.mp4
ffmpeg -i bbb.y4m $A -crf 40 -tile-columns 1 -tile-rows 1 \
  -aom-params num-tile-groups=2 av1_tiles.mp4
ffmpeg -i bbb_odd.y4m $A -crf 40 -aom-params sb-size=128 av1_sb128.mp4
ffmpeg -i bbb.y4m $A -crf 40 -aq-mode variance \
  -aom-params deltaq-mode=1:delta-lf-mode=1 av1_segmentation_deltas.mp4
ffmpeg -i bbb.y4m $A -crf 40 -enable-filter-intra 1 -enable-masked-comp 1 \
  -enable-interinter-wedge 1 -enable-diff-wtd-comp 1 -enable-dist-wtd-comp 1 \
  -enable-dual-filter 1 av1_compound_filters.mp4
ffmpeg -i bbb_odd.y4m -c:v libsvtav1 -preset 6 -crf 40 \
  -svtav1-params superres-mode=1:superres-denom=12 -movflags +faststart av1_superres.mp4
```

libaom's FFmpeg wrapper cannot set super-resolution, so that fixture comes from
SVT-AV1, which pads the odd 250x142 source to a coded 256x144. Each
`av1_*_yuv420.sha256` holds one `FrameDigest` per presentation frame of
FFmpeg/libdav1d's decode of the fixture, hashed exactly as
`big_buck_bunny_av1_yuv420.sha256` above but at each frame's decoded size:

```sh
ffmpeg -c:v libdav1d -i av1_film_grain.mp4 -fps_mode passthrough \
  -f rawvideo -pix_fmt yuv420p ref.yuv
```

libdav1d applies film grain by default, so `av1_film_grain_yuv420.sha256` is
the grain-applied output. `crates/zvidlib-av1-decoder/src/av1_dec/tests.rs` decodes every fixture,
compares every frame with these digests, and checks that each fixture's decode
reaches the tools it is listed against (recorded by `crates/zvidlib-av1-decoder/src/av1_dec/coverage.rs`
in test builds), so a regenerated fixture that stops using a tool fails rather
than silently losing its check. As above, FFmpeg is only the offline fixture
generator and is not a build, test, or runtime dependency.

Each `vp9_bbb_*_yuv420.sha256` holds one `FrameDigest` per presentation
frame of libvpx's own decode of the fixture, hashed as
`big_buck_bunny_av1_yuv420.sha256` above, and each `vp9_bbb_*_rgba.sha256` the
`Rgba8` digests of the same frames after this crate's BT.601
`convert_to_rgba8`, the conversion `native_vp9_video_decoder_factory` applies to
a stream that does not name its colour space:

```sh
ffmpeg -c:v libvpx-vp9 -i vp9_bbb_256x144.mp4 -fps_mode passthrough \
  -f rawvideo -pix_fmt yuv420p ref.yuv
```

`crates/zvidlib-vp9-decoder/src/vp9_dec/tests.rs` compares the decoder's YUV output with the first and
`tests/codec_conformance.rs` and the browser fallback test in
`src/web_decoder.rs` the factory's RGBA output with the second. As above, FFmpeg
and libvpx are only the offline fixture generators and are not build, test, or
runtime dependencies.
