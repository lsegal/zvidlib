# `zvidlib-opus` test fixtures

The fixtures this crate's tests and benchmarks read; other crates that read them reach them here.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`opus_celt_stereo.mp4`, `opus_hybrid_mono.mp4` and `opus_silk_mono.mp4` are
half-second Opus tracks encoded by libopus, one per Opus coding mode: CELT
fullband stereo (TOC configuration 31), hybrid fullband mono (15) and SILK
narrowband mono (1). `crates/zvidlib-opus/tests/opus_codec.rs` checks every packet is in the mode
its name says, so a regenerated fixture that drifts to another mode fails. The
stereo one is music from the bundled `examples/media/BigBuckBunny.mp4`; the
mono ones are a synthetic voice written by `opus_speechlike.py` here, a pulse
train through moving formant filters with an unvoiced burst, which libopus
codes as speech. FFmpeg writes each with a `dOps` box, an edit list whose media
time is the 312-sample pre-skip, and a last sample whose duration is shortened
to trim the end, which is the form zvidlib has to read. They were generated
offline with

```sh
ffmpeg -ss 20 -t 0.5 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ac 2 \
  -ar 48000 -c:a libopus -b:a 96k -frame_duration 20 -movflags +faststart \
  opus_celt_stereo.mp4
python3 opus_speechlike.py speech.f32 1.0
ffmpeg -f f32le -ar 48000 -ac 1 -i speech.f32 -t 0.5 -c:a libopus -b:a 28k \
  -application voip -frame_duration 20 -movflags +faststart opus_hybrid_mono.mp4
ffmpeg -f f32le -ar 48000 -ac 1 -i speech.f32 -t 0.5 -c:a libopus -b:a 8k \
  -application voip -frame_duration 20 -movflags +faststart opus_silk_mono.mp4
```

Each `opus_*_libopus.s16` is libopus's decode of its fixture as interleaved
16-bit little-endian PCM, the first 24 000 frames of

```sh
ffmpeg -c:a libopus -i opus_celt_stereo.mp4 -f s16le opus_celt_stereo.s16
```

FFmpeg trims the pre-skip but, at the version used, not the end padding, which
is why only the encoded length is kept.

`opus_stereo.webm` is the same half second of music encoded by libopus
straight into WebM, which FFmpeg writes with the pre-skip as `CodecDelay` and
the end trim as the last block's `DiscardPadding`.
`opus_stereo_webm_libopus.s16` is libopus's decode of it, which FFmpeg trims
by both to exactly the 24 000 encoded frames:

```sh
ffmpeg -ss 20 -t 0.5 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ac 2 \
  -ar 48000 -c:a libopus -b:a 96k -frame_duration 20 opus_stereo.webm
ffmpeg -c:a libopus -i opus_stereo.webm -f s16le opus_stereo_webm_libopus.s16
```

`opus_stereo.webm` is the same half second of music encoded by libopus
straight into WebM, which FFmpeg writes with the pre-skip as `CodecDelay` and
the end trim as the last block's `DiscardPadding`.
`opus_stereo_webm_libopus.s16` is libopus's decode of it, which FFmpeg trims
by both to exactly the 24 000 encoded frames:

```sh
ffmpeg -ss 20 -t 0.5 -i ../../../../examples/media/BigBuckBunny.mp4 -vn -ac 2 \
  -ar 48000 -c:a libopus -b:a 96k -frame_duration 20 opus_stereo.webm
ffmpeg -c:a libopus -i opus_stereo.webm -f s16le opus_stereo_webm_libopus.s16
``` The official RFC 8251 decoder test
vectors are 75 MB and are not committed; CI downloads them for
`rfc8251_test_vectors_pass_opus_compare`.
