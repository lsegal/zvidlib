# zvidlib

**Frame-exact video and sample-exact audio for Rust and the browser.**

[Website and live demo](https://lsegal.github.io/zvidlib/) ·
[API docs](https://lsegal.github.io/zvidlib/zvidlib/) ·
[crates.io](https://crates.io/crates/zvidlib) ·
[docs.rs](https://docs.rs/zvidlib) ·
[Releases](https://github.com/lsegal/zvidlib/releases/latest)

zvidlib decodes, encodes and muxes video and audio: HEVC, AV1, VP8 and VP9 video and AAC, Opus and Vorbis audio, in MP4 and WebM. It runs natively and in WebAssembly from one API. Ask for frame `n` with `get(n)` and you get exactly frame `n`, even in streams full of B-frames and long groups of pictures. Ask for a sample range and you get exactly those samples. Write frame `n` with `put(n)` and zvidlib encodes it, keeps the audio aligned to it, and muxes the result into a file any player can open.

It's built for apps that treat video as data: editors, compositors, recorders, timeline UIs, thumbnailers, and render pipelines that draw into a GL or WebGL canvas and need the file to match what was drawn.

- **Frame-exact indexed `get` and `put`.** Every read returns the frame you asked for, never the nearest keyframe. zvidlib keeps decoded frames and decoder state between requests, so sequential access stays fast. A seek-preview tier answers scrubs instantly while the exact frame decodes behind it.
- **Sample-exact audio.** AAC, Opus and Vorbis decode and encode alongside the video. Read any exact sample range (`AudioSampleReader` natively, `AudioStream.getRange()` in the browser), or exactly the samples covered by a video frame. Priming, padding and edit lists are handled, so gapless round trips come back sample for sample, and audio is muxed in sync with the video into MP4 and WebM.
- **One API, native and WebAssembly.** The same concepts work in Rust on Windows, macOS and Linux, and in JavaScript through a `wasm-bindgen` package with BigInt-safe frame indices, `Blob` and stream I/O, and stable error codes.
- **Hardware first, pure Rust always.** zvidlib uses NVDEC, Media Foundation (NVENC, Quick Sync, AMF), VideoToolbox and browser WebCodecs when they're available. When they aren't, it falls back to its own pure-Rust codecs, which are checked against libvpx, libdav1d and FFmpeg.
- **No FFmpeg dependency.** No C toolchain, no system codec packages, no GPL. Codecs and containers are Rust crates you can trim down to what your app uses with [Cargo features](#choosing-codecs).

## Built with zvidlib

### [zvid](https://zvid.lsegal.workers.dev)

[zvid](https://github.com/lsegal/zvid) is a collaborative, canvas-based video editor for the browser and desktop (Tauri). Its MP4 export runs on zvidlib: the editor draws each frame onto a canvas, and zvidlib encodes HEVC or AV1 with AAC audio and writes the file, in the browser and in the native app. Its DAW capture plugin, ZVID Capture, records cameras from inside Ableton Live through zvidlib's encoders and muxer.

[Open zvid](https://zvid.lsegal.workers.dev) · [Source](https://github.com/lsegal/zvid)

Building something with zvidlib? Open a pull request to add it here.

## Support

### Video codecs

| Codec | Decode (pure Rust) | Decode (hardware) | Encode (pure Rust) | Encode (hardware) | Browser (`web` feature) |
| --- | --- | --- | --- | --- | --- |
| **HEVC / H.265** | Main, Main 10 | NVDEC (Windows, Linux), Media Foundation (Windows), VideoToolbox (macOS) | Main: lossless, fixed QP or target bitrate, intra-only | Media Foundation: NVENC, Quick Sync, AMF (Windows); VideoToolbox (macOS) | Decode: WebCodecs, falling back to pure Rust. Encode: WebCodecs |
| **AV1** | Main, 8-bit 4:2:0 and monochrome | — | Main, 8-bit monochrome, lossless or quantized | — | Decode: WebCodecs, falling back to pure Rust. Encode: WebCodecs |
| **VP8** | All bitstream versions | NVDEC (Windows, Linux), Media Foundation (Windows) | Key and inter frames, fixed quantizer or target bitrate | — | Decode: WebCodecs, falling back to pure Rust. Encode: WebCodecs |
| **VP9** | Profile 0 | NVDEC (Windows, Linux), Media Foundation (Windows), VideoToolbox (macOS) | Profile 0, key and inter frames | Media Foundation, such as Quick Sync (Windows); VideoToolbox (macOS) | Decode: WebCodecs, falling back to pure Rust. Encode: WebCodecs |

Every decoder returns the same pixels whichever backend runs it. `HardwarePreference::Prefer` picks the fastest backend the host has, `Require` refuses to fall back, and `Avoid` always runs pure Rust. The pure-Rust decoders are checked against reference decoders: bit for bit on libvpx's VP8 and VP9 test vectors, and frame for frame against libdav1d for AV1. The [platform tables](#platform-expectations) list the order each factory tries its backends in.

### Audio codecs

| Codec | Decode (native) | Encode (native) | Decode (browser) | Encode (browser) |
| --- | --- | --- | --- | --- |
| **AAC-LC** | Media Foundation (Windows), AudioToolbox (macOS), Symphonia in pure Rust (Linux and other targets) | Media Foundation (Windows), AudioToolbox (macOS) | WebCodecs | WebCodecs |
| **Opus** | `opus-pure`, a pure-Rust port of libopus | `opus-pure` | WebCodecs, falling back to pure Rust | WebCodecs, falling back to pure Rust |
| **Vorbis** | Pure Rust (Symphonia-derived, up to 8 channels) | Pure-Rust port of libvorbis 1.3.7, bit-identical packets | WebCodecs, falling back to pure Rust | Pure Rust |

### Containers

| Container | Read | Write | Video | Audio |
| --- | --- | --- | --- | --- |
| **MP4** (ISO BMFF) | Ordinary and fragmented files, edit lists, cover art | Seekable MP4 with edit lists, gapless audio trims and generated or supplied cover art | HEVC, AV1, VP9 (`vp09`) | AAC, Opus |
| **WebM** | Seekable files and live `MediaRecorder` captures (unknown-size segments and clusters, every lacing mode) | Seekable WebM with `Cues`, which browsers can seek | AV1, VP8, VP9 | Opus, Vorbis |

zvidlib detects the container from the bytes, never from a file name or MIME type.

### Platforms

| Target | Status |
| --- | --- |
| Windows (CI: x86_64) | Every codec. Hardware: NVDEC (64-bit), Media Foundation (D3D11), NVENC, Quick Sync, AMF |
| macOS (CI: Apple silicon) | Every codec. Hardware: VideoToolbox, AudioToolbox |
| Linux (CI: x86_64) | Every codec except AAC encoding. Hardware: NVDEC (64-bit); encoding is pure Rust |
| Browsers (`wasm32-unknown-unknown`) | Chrome, Edge, Firefox and Safari through WebCodecs, falling back to the pure-Rust decoders. No threads or cross-origin isolation required |

## Quick start

### Rust

```sh
cargo add zvidlib
```

Read exactly frame 400 of a video track:

```rust,ignore
use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, CodecProfile, ColorRange, ExactFrameReader, FrameIndex, HardwarePreference,
    Limits, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, TrackKind, VideoDecoderConfig,
    native_av1_video_decoder_factory,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `pollster` (or any executor) drives zvidlib's async I/O; in-memory sources never wait.
    let source = MemorySource::new(std::fs::read("clip.mp4")?);
    let movie = pollster::block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))?;
    let track = movie.tracks.iter().find(|t| t.kind == TrackKind::Video).ok_or("no video")?;

    let limits = Limits::default();
    let samples = pollster::block_on(track.to_encoded_video_samples(&source, &limits))?;
    let config = VideoDecoderConfig {
        codec: track.codec,
        profile: CodecProfile::Av1Main,
        coded_dimensions: track.dimensions.ok_or("no dimensions")?,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: track.decoder_config.clone(),
    };
    let factory = native_av1_video_decoder_factory();
    let mut reader = ExactFrameReader::new(&factory, config, samples, limits)?;

    // Exactly presentation frame 400, never the nearest keyframe.
    let frame = reader.get(FrameIndex(400), &CancellationToken::new())?;
    println!("{}x{} {:?}", frame.dimensions.width, frame.dimensions.height, frame.pixel_format);
    Ok(())
}
```

Encode 90 RGBA frames to a VP9 MP4, in hardware where the host has a VP9 encoder:

```rust,ignore
use zvidlib::io::MemorySink;
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Plane, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_vp9_video_encoder_factory,
};
use pollster::block_on;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = Limits::default();
    let size = VideoDimensions::new(640, 360, &limits)?;
    let mut encoder = native_vp9_video_encoder_factory().create(
        &VideoEncoderConfig {
            codec: Codec::Vp9,
            profile: CodecProfile::Vp9Profile0,
            coded_dimensions: size,
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Prefer,
            timescale: 30,
            frame_duration: 1, // 30 fps
            configuration: Vec::new(),
        },
        &limits,
    )?;
    let track = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Video(size),
    };
    let mut muxer = block_on(Mp4Muxer::new(MemorySink::new(), vec![track], 1_000))?;

    for n in 0..90_u64 {
        let rgba = vec![(n * 3) as u8; 640 * 360 * 4]; // your renderer's pixels
        let plane = Plane { data: rgba, stride: 640 * 4 };
        let frame = VideoFrame::new(size, PixelFormat::Rgba8, ColorRange::Limited, vec![plane], &limits)?;
        let source = FrameSource::Cpu(CpuFrameSource { frame: &frame, orientation: Orientation::TopLeft });
        for sample in block_on(encoder.encode(FrameIndex(n), source))? {
            block_on(muxer.write_sample(0, sample))?;
        }
    }
    for sample in block_on(encoder.finish())? {
        block_on(muxer.write_sample(0, sample))?;
    }
    std::fs::write("out.mp4", block_on(muxer.finish())?.into_inner())?;
    Ok(())
}
```

`MediaOutput` wraps the same encoder and muxer contracts for synchronized video and audio, enforcing consecutive `put_video(n)` and `put_audio(n)` indices and frame-aligned audio. `WebmDemuxer` and `WebmMuxer` are drop-in replacements for the MP4 types. [`examples/`](examples/README.md) has runnable native OpenGL playback with audio, scrubbing and seek previews.

### JavaScript

Download `zvidlib-web-vX.Y.Z.tgz` from the [latest release](https://github.com/lsegal/zvidlib/releases/latest) and install it:

```sh
npm install ./zvidlib-web-vX.Y.Z.tgz
```

Open a file and read an exact frame:

```js
import init, { MediaInput, errorCode } from "zvidlib";

await init();

const input = await MediaInput.open(await (await fetch("clip.mp4")).blob());
const video = input.video(0);
try {
  const frame = await video.get(400n); // BigInt frame index, exact
  ctx.putImageData(new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height), 0, 0);
} catch (error) {
  console.log(errorCode(error)); // e.g. "UNSUPPORTED" when no decoder can handle the track
}
input.close();
```

Play and seek a remote MP4 without downloading it: `OnDemandPlayback` reads only the movie header and the compressed samples playback reaches, with HTTP range requests (or from a `Blob`), into byte-budgeted caches. No call waits on the network; one that needs a sample not loaded yet throws `WOULD_BLOCK`, and `prefetch()` loads it:

```js
import init, { OnDemandPlayback, errorCode } from "zvidlib";

await init();

const playback = await OnDemandPlayback.open("https://example.com/clip.mp4", {
  audioContext: new AudioContext(), // its clock times playback: create it in a click handler
  videoBudgetBytes: 16 * 1024 * 1024,
});
function render() {
  try {
    if (!playback.isPlaying) playback.play();
    const { picture } = playback.present(); // null until the clock reaches a new frame
    if (picture) ctx.putImageData(new ImageData(new Uint8ClampedArray(picture.pixels), picture.width, picture.height), 0, 0);
  } catch (error) {
    if (errorCode(error) !== "WOULD_BLOCK") throw error;
    playback.prefetch();
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
```

The server must answer range requests and, cross-origin, expose `Content-Range` through CORS. Video decodes on zvidlib's software decoders, AAC audio through WebCodecs, and Opus audio through WebCodecs or zvidlib's software decoder. A video with no audio track needs no `audioContext`: it plays on `performance.now()`, or on the context's clock when one is given.

Record a canvas to WebM (or `"mp4"`) with synchronized Opus audio:

```js
import init, { AudioBuffer, CreateOptions, MediaOutput, SampleRange, VideoFrame } from "zvidlib";

await init();

const options = new CreateOptions("webm");
options.videoCodec = "vp9"; // "av1", "vp8" or "vp9" for WebM; "av1", "hevc" or "vp9" for MP4
options.setTimeline(30, 1, 48_000); // 30 fps video, 48 kHz audio
const output = await MediaOutput.create(options);

for (let n = 0n; n < 90n; n++) {
  draw(ctx, n);
  const { data } = ctx.getImageData(0, 0, width, height);
  await output.video(0).put(n, VideoFrame.rgba(width, height, data));
  const start = n * 1600n; // 48 000 / 30 samples per frame
  await output.audio(0).put(n, new AudioBuffer(new SampleRange(start, start + 1600n), 48_000, 2, pcm(start)));
}

const blob = await output.finish(); // Blob { type: "video/webm" }
```

The [landing page demo](https://lsegal.github.io/zvidlib/#demo) does both live in your browser, and [`examples/web_canvas`](examples/web_canvas) and [`examples/web_encode`](examples/web_encode) are complete WebGL playback and encoding pages.

## Reference

### Data paths

Video frames support two families of destination and source:

- CPU-backed planes with explicit pixel format, dimensions, stride, color space, and transfer metadata.
- GPU-backed images associated with a GL texture or framebuffer on native targets, and a WebGL texture or framebuffer in browsers.

GPU interop is capability-based. A backend may share an image without a copy, copy on the GPU, or fall back through CPU memory. Callers can require a particular behavior and receive an unsupported-capability error instead of an implicit expensive fallback.

Audio buffers carry their sample format, channel layout, sample rate, exact timeline interval, and any priming or padding information needed for gapless synchronization.

### Platform expectations

| Capability | Native | WebAssembly/browser |
| --- | --- | --- |
| Input/output | Files, memory, caller storage | `Blob`, streams, memory, File System Access handles when supplied |
| Video acceleration | Pluggable software or platform backend | WebCodecs when available, otherwise the pure-Rust software decoder |
| Graphics | OpenGL-family context supplied by caller | WebGL context supplied by caller |
| Audio | Raw buffers and default-device PCM output | Web Audio buffers/nodes supplied by caller |
| Concurrency | Worker threads where safe | Async tasks; workers/threads only when browser isolation permits |

Codec and hardware availability varies by browser and operating system. Opening media reports capabilities explicitly, so supporting a container doesn't mean every platform can encode or decode every codec it carries.

The native HEVC Main backends each factory can select, in the order it tries them. `Avoid` always takes the pure-Rust backend; `Require` takes only the hardware ones.

| HEVC Main | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Decode (`native_hevc_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | VideoToolbox, then pure Rust | NVDEC (64-bit), then pure Rust |
| Encode (`native_hevc_video_encoder_factory`) | Media Foundation hardware (NVENC, Quick Sync, AMF), then Media Foundation software, then pure Rust | VideoToolbox hardware, then pure Rust | pure Rust |

The HEVC decoder also takes HEVC Main 10 (`CodecProfile::HevcMain10`, 8- to 10-bit 4:2:0) on every platform, always in pure Rust.

| Decode | Windows | macOS | Linux |
| --- | --- | --- | --- |
| VP8 (`native_vp8_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | pure Rust | NVDEC (64-bit), then pure Rust |
| VP9 profile 0 (`native_vp9_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | VideoToolbox, then pure Rust | NVDEC (64-bit), then pure Rust |
| AV1 Main (`native_av1_video_decoder_factory`) | pure Rust | pure Rust | pure Rust |

Every native encoder, by operating system, and what does the encoding. A *hardware* encoder runs on the GPU or media engine. A *platform* encoder is the operating system's own codec; it reports `CodecImplementation::Hardware` because zvidlib runs none of the codec itself. A *software* encoder is zvidlib's own pure-Rust code and reports `CodecImplementation::Software`. `VideoEncoder::implementation()` and `backend_name()` report which one a created video encoder is.

| Encoder | Windows | macOS | Linux |
| --- | --- | --- | --- |
| HEVC Main (`native_hevc_video_encoder_factory`) | **Hardware**: Media Foundation (NVENC, Quick Sync, AMF) for `Prefer`/`Require` target-bitrate configurations, with `Prefer` falling back to Media Foundation's **software** HEVC encoder. **Software**: pure Rust for `Avoid`, lossless, fixed QP, and the last `Prefer` fallback | **Hardware**: VideoToolbox for `Prefer`/`Require` target-bitrate configurations (`Rgba8`/`Bgra8`, even dimensions). **Software**: pure Rust for `Avoid`, lossless, fixed QP, and the `Prefer` fallback | **Software**: pure Rust; `Require` reports `HardwareUnavailable` |
| AV1 Main, 8-bit monochrome (`native_av1_video_encoder_factory`) | **Software**: pure Rust | **Software**: pure Rust | **Software**: pure Rust |
| VP8 (`native_vp8_video_encoder_factory`) | **Software**: pure Rust; `Require` reports `HardwareUnavailable` | **Software**: pure Rust; `Require` reports `HardwareUnavailable` | **Software**: pure Rust; `Require` reports `HardwareUnavailable` |
| VP9 profile 0 (`native_vp9_video_encoder_factory`) | **Hardware**: Media Foundation (for example Quick Sync) for `Prefer`/`Require` where the adapter has a VP9 encoder. **Software**: pure Rust for `Avoid`, and for `Prefer` without one | **Hardware**: VideoToolbox for `Prefer`/`Require` on Macs with a VP9 encoder. **Software**: pure Rust for `Avoid`, and for `Prefer` without one | **Software**: pure Rust; `Require` reports `HardwareUnavailable` |
| AAC-LC (`native_aac_audio_encoder_factory`) | **Platform**: Media Foundation AAC encoder MFT (44.1 or 48 kHz, mono or stereo) | **Platform**: AudioToolbox | None: reports `HardwareUnavailable` |
| Opus (`native_opus_audio_encoder_factory`) | **Software**: `opus-pure`, a pure-Rust port of libopus (48 kHz, mono or stereo) | **Software**: `opus-pure` | **Software**: `opus-pure` |
| Vorbis (`native_vorbis_audio_encoder_factory`) | **Software**: zvidlib's port of the libvorbis 1.3.7 encoder (mono or stereo, every rate libvorbis supports) | **Software**: the libvorbis port | **Software**: the libvorbis port |

| Audio decoder | Windows | macOS | Linux |
| --- | --- | --- | --- |
| AAC-LC (`NativeAacDecoder`) | **Platform**: Media Foundation AAC decoder MFT (mono or stereo) | **Platform**: AudioToolbox (mono or stereo) | **Software**: Symphonia's AAC decoder (mono or stereo) |
| Opus (`NativeOpusDecoder`) | **Software**: `opus-pure` | **Software**: `opus-pure` | **Software**: `opus-pure` |
| Vorbis (`NativeVorbisDecoder`) | **Software**: vendored Symphonia Vorbis decoder | **Software**: vendored Symphonia Vorbis decoder | **Software**: vendored Symphonia Vorbis decoder |

What each backend accepts, how it is verified, and the full browser API surface are described in the [backend and API reference](ARCHITECTURE.md#appendix-backend-and-api-reference) at the end of `ARCHITECTURE.md`.

### Using a release

zvidlib is published to [crates.io](https://crates.io/crates/zvidlib), with API documentation on
[docs.rs](https://docs.rs/zvidlib). For a native build, depend on the `zvidlib` crate:

```toml
[dependencies]
zvidlib = { version = "<version>" }
```

with the latest version from crates.io in place of `<version>`, or run
`cargo add zvidlib`. The workspace's `zvidlib-*` crates are published
alongside `zvidlib` at the same version, but they are implementation crates that `zvidlib`
re-exports; depend on `zvidlib` only. The native API requires Rust 1.85 or later, as recorded by
`rust-version` in `Cargo.toml`; platform codec adapters retain their documented platform capability
checks.

To pin a release tag, or to use a commit that is not yet on crates.io, use a Git dependency
instead, with the release version in place of `X.Y.Z`:

```toml
[dependencies]
zvidlib = { git = "https://github.com/lsegal/zvidlib.git", tag = "vX.Y.Z" }
```

Each GitHub release also attaches the `.crate` archives it published, such as `zvidlib-X.Y.Z.crate`,
for inspection or offline packaging.

#### Choosing codecs

Every native codec is a Cargo feature, one per codec and direction, and the
default features enable all of them along with the platform hardware backends.
An application that needs only some codecs turns the defaults off and names the
ones it uses, and builds none of the others:

```toml
[dependencies]
# Decode VP9 only, in software.
zvidlib = { version = "<version>", default-features = false, features = ["vp9-decoder"] }
# Both AV1 directions and the HEVC encoder, with hardware acceleration.
zvidlib = { version = "<version>", default-features = false, features = ["av1", "hevc-encoder", "hardware"] }
```

| Codec | Decoder | Encoder | Both |
| --- | --- | --- | --- |
| AV1 | `av1-decoder` | `av1-encoder` | `av1` |
| VP8 | `vp8-decoder` | `vp8-encoder` | `vp8` |
| VP9 | `vp9-decoder` | `vp9-encoder` | `vp9` |
| HEVC | `hevc-decoder` | `hevc-encoder` | `hevc` |
| Opus | `opus-decoder` | `opus-encoder` | `opus` |
| Vorbis | `vorbis-decoder` | `vorbis-encoder` | `vorbis` |
| AAC | `aac-decoder` | `aac-encoder` | `aac` |

- `all` enables every codec feature in the table. The default features are
  `all` and `hardware`.
- `hardware` builds the platform backends (NVDEC, Media Foundation,
  VideoToolbox) for whichever enabled codecs have one. Without it the enabled
  codecs are pure Rust: `HardwarePreference::Require` reports
  `HardwareUnavailable`, and `Prefer` falls back to software, as it does on a
  host with no backend.
- Features are additive and mix freely. The containers, the codec traits and
  the configuration records the containers read and write (`OpusHead`,
  `VorbisConfig`, the AV1, VP9 and HEVC syntax) are built in every
  configuration, so MP4 and WebM files with any track open regardless; a track
  whose codec is not enabled has no native decoder or encoder.
- The AAC decoder runs on the operating system's decoder on macOS and Windows
  and on Symphonia's elsewhere; the AAC encoder needs macOS or Windows.
- In a `web` build, a codec whose feature is off has no software fallback
  behind `WebCodecs`.
- `native` gates no library code. It only keeps the native-only examples out of
  `wasm32` builds, so applications do not need it for the native codecs.

#### Browser package

The browser package is not on crates.io or npm: it is a GitHub release asset. For a browser build,
download the `zvidlib-web-vX.Y.Z.tgz` asset from the
[latest GitHub release](https://github.com/lsegal/zvidlib/releases/latest) and install it:

```sh
npm install ./zvidlib-web-vX.Y.Z.tgz
```

The package contains `zvidlib.js`, `zvidlib.d.ts`, and `zvidlib_bg.wasm` generated by
`wasm-pack --target web`; import it as an ES module and call `await init()` once before using an
exported class. Browser and Cargo consumers must use the same release version when they exchange
media or API contracts.

#### Versioning

zvidlib follows pre-1.0 semantic versioning: patch releases preserve documented public API and
behavior, while a minor release may make breaking API changes. Every release publishes its Rust
crates to crates.io and attaches its browser package to the GitHub release; release notes call out
any platform capability change. The crates are uploaded after the GitHub release is created and
paced to crates.io's rate limits, so a release whose crates are new names can take a few hours to
appear on crates.io.

### Building the library

Install stable Rust with the `wasm32-unknown-unknown` target, then run:

```console
cargo check --workspace --features native
cargo check --target wasm32-unknown-unknown --features web
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features native -- -D warnings
```

These commands validate the library on both targets. The repository is a
Cargo workspace: `zvidlib` at the root, which is the only crate to depend on,
and the codec, container and shared-core crates it is built from under
`crates/` (see `ARCHITECTURE.md`). Without `--workspace` or `-p <crate>`, a
cargo command at the root covers the root package only; `cargo test -p
zvidlib-vp8` runs one crate's tests, `cargo test --workspace --features native`
every package's.

The VP9 decoder is additionally checked against every VP9 profile 0 test vector
libvpx's own test suite decodes, which CI downloads from the WebM project. To
run that check locally, download the vectors named in
`crates/zvidlib-vp9-decoder/tests/fixtures/libvpx_vp9_test_vectors.txt`, each with its `.md5` file,
from `https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`
into one directory, then:

```console
ZVIDLIB_VP9_VECTORS=/path/to/vectors cargo test -p zvidlib-vp9-decoder --lib vp9_dec::tests::libvpx_test_vectors -- --ignored
```

### Building, testing, and using WebAssembly

The `web` feature excludes native-only integrations. Install the Rust target and verify it directly:

```console
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --features web
```

Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) and create a browser-ready ES module package:

```console
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --features web
python -m http.server 8000
```

The build creates `pkg/zvidlib.js`, `pkg/zvidlib.d.ts`, `pkg/zvidlib_bg.wasm`, and package metadata. Open `http://localhost:8000/` and import the generated module with `import init, { ... } from "./pkg/zvidlib.js"`. Call `await init()` exactly once before using an exported class. Deploy the generated JavaScript and WASM files together, with the server returning `application/wasm` for the `.wasm` file. Browser security rules require HTTP(S); opening a page as a `file:` URL will not work reliably.

For the browser examples, `examples/web_canvas/package.json` and `examples/web_encode/package.json` wrap the `wasm-pack build` and serving steps above into one `pnpm dev` command; see [examples/README.md](examples/README.md#web-canvas-web_canvas).

Run the browser integration suite in an installed Chrome browser with:

```console
wasm-pack test --headless --chrome --features web
```

The suite verifies Blob and stream input, cancellation, reader-lock cleanup, BigInt range handling, typed-array copy lifetimes, browser-object ownership, stable errors, Blob output, decoding the bundled HEVC sample and a VP9 track through WebCodecs, and the software fallback for browsers whose WebCodecs cannot decode a track. The base build does not require WASM threads or cross-origin isolation.

The `web` feature's video decoder uses `web-sys`'s `WebCodecs` bindings, which that crate gates behind `--cfg=web_sys_unstable_apis` because the spec is still evolving. `.cargo/config.toml` sets that flag for the `wasm32-unknown-unknown` target automatically, so the `cargo`/`wasm-pack` commands above need no extra flags.

### Website

The [project website](https://lsegal.github.io/zvidlib/) lives in [`site/`](site/). The Docs workflow builds it with the WebAssembly package and the rustdoc API documentation into one GitHub Pages artifact and deploys it on every push to `main`. The API documentation stays at [`/zvidlib/`](https://lsegal.github.io/zvidlib/zvidlib/), with [`/api/`](https://lsegal.github.io/zvidlib/api/) redirecting to it. To preview the site locally, build the package into `site/pkg` and serve the directory:

```console
wasm-pack build --target web --out-dir site/pkg --features web
mkdir -p site/media && cp examples/media/BigBuckBunny.av1.mp4 site/media/
python -m http.server 8000 --directory site
```

### Repository layout

`src/` is the `zvidlib` crate: the public API, synchronized output, playback control, the seek-preview tier, and the browser (`web`) boundary. `crates/` holds the shared core (errors, timeline, media values, I/O, codec traits), the MP4 and WebM containers, one crate per codec and direction, and the platform hardware backends. `examples/` has runnable native and browser examples, `benches/` the criterion benchmarks, and `site/` the project website. [ARCHITECTURE.md](ARCHITECTURE.md) describes component boundaries, seeking and caching behavior, container responsibilities, and portability constraints.

## Roadmap

zvidlib's core is complete: exact reads, synchronized writes, two containers, seven codecs, and hardware backends on every desktop platform. The remaining work is breadth and polish:

1. **Color AV1 encoding.** The native AV1 encoder writes 8-bit monochrome key frames. Color and inter-frame AV1 come next. Browsers already encode color AV1 through WebCodecs.
2. **Inter-frame HEVC in pure Rust.** The software HEVC encoder writes every picture as an IDR. Hardware HEVC encoders already use inter prediction.
3. **Playback backends.** A browser `Playback` backend over a `MediaInput` (`Playback.play()` and `present()` report `UNSUPPORTED` today; `OnDemandPlayback` plays a URL or `Blob` read on demand, and [`examples/web_canvas`](examples/web_canvas) shows the equivalent loop built from `get(n)` and Web Audio), WebCodecs video decoding for on-demand playback, and portable native audio devices beyond default-device PCM output.
4. **More hardware.** Hardware encoders on Linux, hardware HEVC Main 10 decoding, and hardware AV1 decoding.
5. **Fragmented MP4 writing** for streaming and crash-safe recording.
6. **WebM VP8 alternate-reference frames.** A VP8 frame the encoder hid as a separate block is currently indexed as a frame of its own, and reading it fails.
7. **npm package.** Publish the browser package to npm alongside the GitHub release asset.
8. **1.0.** Stabilize the Rust and JavaScript APIs after cross-platform conformance testing.

## License

zvidlib is released under the [MIT License](LICENSE). Third-party code it vendors or ports is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
