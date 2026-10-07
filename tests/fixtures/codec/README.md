# Codec test fixtures

The fixtures of the codecs zvidlib implements live with the crate that implements each one (#604), each described by that crate's `tests/fixtures/README.md`:

- `crates/zvidlib-av1/tests/fixtures/`
- `crates/zvidlib-av1-decoder/tests/fixtures/`
- `crates/zvidlib-hevc-decoder/tests/fixtures/`
- `crates/zvidlib-opus/tests/fixtures/`
- `crates/zvidlib-vorbis-decoder/tests/fixtures/`
- `crates/zvidlib-vp8/tests/fixtures/`
- `crates/zvidlib-vp9-decoder/tests/fixtures/`

This directory keeps the AAC fixtures, which the root package's `NativeAacDecoder` tests and audio benchmarks read.

FFmpeg is only the offline fixture oracle. It is not a build, test, or runtime
dependency, and zvidlib's native codec implementations may not call or link it.

`aac_lc_mono_48k.m4a` is a two-second 48 kHz mono AAC-LC track used by the
`audio_decode` benchmark target. The bundled `examples/media/BigBuckBunny.mp4`
sample supplies the stereo arm, but it is stereo-only and carries no edit list,
while `NativeAacDecoder` accepts AAC-LC mono as well and rejects everything
beyond stereo - so mono and stereo together are the backend's entire supported
input space. This fixture also carries a real `elst` whose media time is the
decoder priming, which is what makes the edit-list and gapless priming/padding
mapping in `AacSampleReader` measurable at all. It was generated offline with

```sh
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=2.048" \
  -af "volume=0.5,tremolo=f=5:d=0.7" -ac 1 -c:a aac -b:a 64k \
  -movflags +faststart aac_lc_mono_48k.m4a
```

The duration is 2.048 s rather than a round two seconds so that the media
duration is an exact multiple of the 1024-sample AAC-LC access unit and no
packet's indexed interval is truncated. As above, FFmpeg is only the offline
fixture generator and is not a build, test, or runtime dependency.

`aac_reference.bin` is the AAC decode both AAC tracks above produced before
issue #561 replaced Symphonia's AAC decoder with the platform ones on macOS
and Windows, and what `tests/native_aac_decoder.rs` holds AudioToolbox and
Media Foundation to; on Linux, which still decodes through Symphonia, the same
test holds Symphonia to it. For the bundled stereo sample and then the mono
fixture, it holds three 1024-frame windows of each one's presentation
timeline, as `AacSampleReader` returned them over `symphonia-codec-aac` 0.5.5
with a preroll of two access units: the first window, one starting at half the
presentation length plus 333 frames, and the last full window. The samples are
interleaved little-endian `f32`, 36,864 bytes in all. It must only ever be
regenerated with Symphonia, on Linux; a platform decoder is compared against it
by signal-to-noise ratio, since AAC decoders are not bit-exact with each other.
