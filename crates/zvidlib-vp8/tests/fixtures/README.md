# `zvidlib-vp8` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`vp8/vp80-00-comprehensive-001.ivf` to `-018.ivf` are libvpx's VP8 decoder
test vectors, downloaded unmodified from
`https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`
together with the `.ivf.md5` file libvpx publishes beside each. Every line of
an `.md5` file is the MD5 of one shown frame's I420 output from libvpx,
cropped to the display size. They are the set libvpx checks its own VP8
decoder against, and include every bitstream version (003 and 007 are version
1, 004 version 2 and 005 version 3, so bilinear and full-pixel chroma
prediction), odd frame sizes (006 and 014 are 175x143), a 1432x888 stream
(008), streams with up to four key frames, and a hidden key frame (018).
`crates/zvidlib-vp8/src/tests.rs` decodes each vector and compares every shown frame with
libvpx's digest; `tests/vp8_conformance.rs` checks that every frame comes
back identically through `ExactFrameReader` in sequential, reverse and
alternating order. They are distributed under the libvpx license; see
`THIRD_PARTY_NOTICES.md`.

`vp8/vp8_testsrc2_98x66.webm` is a 30-frame VP8 WebM track with a key frame
every 12 frames and a size that is not a whole number of macroblocks, and
`vp8_testsrc2_98x66.webm.md5` is libvpx's decode of it, one MD5 of each
frame's I420 output per line. They were generated offline with

```sh
ffmpeg -f lavfi -i "testsrc2=size=98x66:rate=25" -t 1.2 -c:v libvpx -g 12 \
  -auto-alt-ref 0 -b:v 300k -deadline good vp8_testsrc2_98x66.webm
ffmpeg -c:v libvpx -i vp8_testsrc2_98x66.webm -f framemd5 - | grep -v '^#' \
  | awk -F', *' '{print $6}' > vp8_testsrc2_98x66.webm.md5
```

`crates/zvidlib-vp8/src/tests.rs` demuxes it with `WebmDemuxer` and compares every frame with
libvpx's digest, and `tests/vp8_conformance.rs` reads it back through
`ExactFrameReader` in sequential, reverse and alternating order. As above,
FFmpeg is only the offline fixture generator.

`vp8/vp8_altref_98x66.ivf` is a 42-frame VP8 IVF stream from a two-pass
libvpx encode with automatic alternate references, so frames 1 and 17 are
hidden inter frames that only update the alternate reference, and 40 frames
are shown. `vp8_altref_98x66.ivf.md5` is libvpx's decode of it, one MD5 of
each shown frame's I420 output per line. None of libvpx's own vectors has a
hidden alternate reference, which is what this covers. They were generated
offline with

```sh
ffmpeg -f lavfi -i "testsrc2=size=98x66:rate=30:duration=1.3333334" \
  -pix_fmt yuv420p src98.y4m
for pass in 1 2; do
  ffmpeg -y -i src98.y4m -c:v libvpx -auto-alt-ref 1 -lag-in-frames 16 \
    -arnr-maxframes 7 -arnr-strength 5 -g 40 -b:v 200k -pass $pass \
    -passlogfile vp8pl -f ivf vp8_altref_98x66.ivf
done
ffmpeg -c:v libvpx -i vp8_altref_98x66.ivf -f framemd5 - | grep -v '^#' \
  | awk -F', *' '{print $6}' > vp8_altref_98x66.ivf.md5
```

`crates/zvidlib-vp8/src/tests.rs` compares every shown frame with libvpx's digest, and
`tests/vp8_conformance.rs` holds a hardware VP8 decoder, where the host has
one, to the software decoder's output and exact-frame seeks on it.

FFmpeg's own Vorbis decode is not used for these references: at the version
used it did not trim to the granule positions consistently.
`vp8/vp8_altref_98x66.webm` is a two-pass VP8 WebM encode of `testsrc2` with
alternate references: 43 WebM blocks, three of which are hidden alternate
reference frames (`show_frame` = 0) stored as blocks of their own, at the
timestamp of the shown frame after them. `vp8_altref_98x66.webm.md5` is
libvpx's decode of its 40 shown frames, one MD5 of each frame's I420 output per
line. They were generated offline with

```sh
for pass in 1 2; do
  ffmpeg -f lavfi -i "testsrc2=size=98x66:rate=25" -t 1.6 -c:v libvpx -g 16 \
    -auto-alt-ref 1 -lag-in-frames 16 -b:v 300k -deadline good -pass $pass \
    vp8_altref_98x66.webm
done
ffmpeg -c:v libvpx -i vp8_altref_98x66.webm -f framemd5 - | grep -v '^#' \
  | awk -F', *' '{print $6}' > vp8_altref_98x66.webm.md5
```

(the first pass may write to the null muxer instead). `crates/zvidlib-vp8/src/tests.rs`
checks that `WebmDemuxer` presents only the shown frames and that each matches
libvpx's digest, and `tests/vp8_conformance.rs` and the browser tests in
`src/web_decoder.rs` read every shown frame back in sequential, reverse and
alternating order, decoding through the hidden ones.
