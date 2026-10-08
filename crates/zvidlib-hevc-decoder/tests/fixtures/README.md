# `zvidlib-hevc-decoder` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`big_buck_bunny_hevc_rgba.sha256` contains canonical `FrameDigest` values for
all 768 presentation frames in the repository's existing
`examples/media/BigBuckBunny.mp4` HEVC Main sample (1920x1080, 32 seconds).
The reference RGBA frames were decoded with FFmpeg 6.0 (forcing the BT.601
matrix regardless of resolution, matching the fixed BT.601 coefficients
`picture_to_rgba` in `crates/zvidlib-hevc-decoder/src/lib.rs` always uses), then fingerprinted with
the dimensions, RGBA8 format, limited color range, plane count, and active
pixels defined by `FrameDigest::from_frame`.

`bbb_hevc_512x288_gop768.mp4` and `bbb_hevc_512x288_gop32.mp4` are the paired
random-access-cadence tracks the `exact_seek` benchmark target measures over.
Both are the same 768 frames of the bundled `examples/media/BigBuckBunny.mp4`
sample re-encoded at 512x288 with the same encoder, preset and quality; the
*only* difference between them is `keyint`, so the first carries one
random-access point and the second twenty-four. That is what makes the gap
between them attributable to the cadence rather than to resolution, bitrate or
content, which the bundled sample alone cannot do: it codes its 768 frames as a
single group of pictures and can therefore only ever describe the worst case.
They were generated offline with

```sh
for g in 768 32; do
  ffmpeg -i ../../../../examples/media/BigBuckBunny.mp4 -an -vf scale=512:288 \
    -c:v libx265 -preset medium -crf 32 \
    -x265-params "keyint=$g:min-keyint=$g:scenecut=0" \
    -tag:v hvc1 -movflags +faststart "bbb_hevc_512x288_gop${g}.mp4"
done
```

`scenecut=0` is what makes the cadence exact rather than approximate: without
it x265 inserts extra key frames wherever the content changes, and the
`keyint=768` track would not have had one random-access point. As above, FFmpeg
is only the offline fixture generator and is not a build, test, or runtime
dependency. 512x288 keeps the pair under a megabyte together; the absolute
per-frame decode cost at that size is smaller than the bundled 1080p sample's,
but the ratio between the two arms - which is what the cadence question is
about - is not affected by it.

`bbb_hevc_512x288_gop32_rgba.sha256` carries the canonical `FrameDigest` of
each of the 768 presentation frames of `bbb_hevc_512x288_gop32.mp4`, in the
same format as `big_buck_bunny_hevc_rgba.sha256`. x265 codes that track as
open groups of pictures: each CRA picture after the first is followed in decode
order by RASL pictures that are presented before it and reference pictures from
the group before it, which is what the issue #506 regression test in
`crates/zvidlib-hevc-decoder/src/lib.rs` needs. The reference frames were decoded with FFmpeg 6.0 and
converted with the same BT.601 matrix:

```sh
ffmpeg -i bbb_hevc_512x288_gop32.mp4   -vf "scale=in_color_matrix=bt601:in_range=tv" -pix_fmt rgba -f rawvideo   bbb_hevc_512x288_gop32.rgba
```

then fingerprinted 512x288x4 bytes at a time with `FrameDigest::from_frame`
over a limited-range `Rgba8` frame. The same command reproduces the first
frames of `big_buck_bunny_hevc_rgba.sha256` from the bundled sample.

`bbb_hevc_main10_128x72.mp4` is a 12-frame HEVC Main 10 (`yuv420p10le`) track
cut from the bundled `examples/media/BigBuckBunny.mp4` sample, with one
random-access point and B-frames, for the Main 10 software decode (issue #508).
It was generated offline with

```sh
ffmpeg -ss 10 -i ../../../../examples/media/BigBuckBunny.mp4 -an -frames:v 12   -vf scale=128:72 -pix_fmt yuv420p10le -c:v libx265 -profile:v main10   -preset medium -crf 30   -x265-params "keyint=12:min-keyint=12:scenecut=0:bframes=3"   -tag:v hvc1 -movflags +faststart bbb_hevc_main10_128x72.mp4
```

`bbb_hevc_main10_128x72_yuv420p10le.sha256` is the SHA-256 of each
presentation frame of FFmpeg 6.0's own decode of that track
(`ffmpeg -i bbb_hevc_main10_128x72.mp4 -f rawvideo -pix_fmt yuv420p10le`),
the planar little-endian 16-bit layout `Picture::to_planar_le16` produces. It
is the independent reference: the decoder's 10-bit samples must equal
FFmpeg's before any color conversion.

`bbb_hevc_main10_128x72_rgba.sha256` carries one canonical `FrameDigest` per
presentation frame of the same track's `Rgba8` output. FFmpeg's scaler has no
10-bit path that reproduces the fixed-point BT.601 matrix `picture_to_rgba`
uses (its 8-bit fast path is the one the Main fixture above matches), so these
digests were computed by applying that matrix, as documented in
`crates/zvidlib-color/src/color_convert.rs`, to FFmpeg's `yuv420p10le` decode in a separate
Python script, with each 10-bit sample shifted up by one bit to the matrix's
11-bit scale. As above, FFmpeg is only the offline fixture generator and is not
a build, test, or runtime dependency.
