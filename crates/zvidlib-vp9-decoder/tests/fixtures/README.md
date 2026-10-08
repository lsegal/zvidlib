# `zvidlib-vp9-decoder` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`vp9_bbb_256x144.mp4` and `vp9_bbb_250x142.mp4` are VP9 profile 0 (8-bit
4:2:0) tracks encoded by libvpx from the bundled sample, for the VP9 decoder
(issue #527). The first is 48 frames at 256x144 in two groups of pictures,
encoded in two passes so that libvpx codes hidden alternate reference frames,
each carried in a superframe with the frame shown after it; the second is 12
frames at 250x142, whose edges are not a multiple of 8. Both leave their color
space unspecified and use studio range. They were generated offline with
FFmpeg 6.0's libvpx wrapper:

```sh
B=../../../../examples/media/BigBuckBunny.mp4
ffmpeg -ss 10 -i $B -an -frames:v 48 -vf scale=256:144 -pix_fmt yuv420p bbb.y4m
ffmpeg -ss 20 -i $B -an -frames:v 12 -vf scale=250:142 -pix_fmt yuv420p bbb_odd.y4m
A="-c:v libvpx-vp9 -auto-alt-ref 1 -deadline good -cpu-used 2 -row-mt 0"
W="$A -b:v 200k -g 24 -keyint_min 24 -lag-in-frames 16 -arnr-maxframes 5"
O="$A -b:v 150k -g 12 -keyint_min 12 -lag-in-frames 8"
ffmpeg -i bbb.y4m $W -pass 1 -passlogfile a -f null /dev/null
ffmpeg -i bbb.y4m $W -pass 2 -passlogfile a -movflags +faststart vp9_bbb_256x144.mp4
ffmpeg -i bbb_odd.y4m $O -pass 1 -passlogfile b -f null /dev/null
ffmpeg -i bbb_odd.y4m $O -pass 2 -passlogfile b -movflags +faststart vp9_bbb_250x142.mp4
```

`libvpx_vp9_test_vectors.txt` lists the VP9 profile 0 test vectors of libvpx's
own test suite (the `vp90-2-*` names of libvpx's `test/test_vectors.cc`).
They are not checked in: CI downloads each, with the `.md5` file of per-frame
digests libvpx's test harness checks, from the WebM project's test-data bucket
and runs the ignored `vp9_dec::tests::libvpx_test_vectors` test over them, which
reads the WebM and IVF files directly and requires every shown frame's MD5 to
match libvpx's.

`vp9_bbb_256x144.webm` is `vp9_bbb_256x144.mp4` remuxed to WebM without
re-encoding (`ffmpeg -i vp9_bbb_256x144.mp4 -c copy vp9_bbb_256x144.webm`), a
`V_VP9` track with no `CodecPrivate`; `crates/zvidlib-vp9-decoder/src/vp9_dec/tests.rs` checks that
`WebmDemuxer` yields the same chunks and that they decode to the same digests.
