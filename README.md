# zvidlib

zvidlib is a planned Rust library for frame-accurate video and synchronized audio I/O on native and WebAssembly targets. Its primary jobs are reading an MP4 into a GL/WebGL canvas and writing canvas frames plus an audio stream into an MP4, behind a small API centered on indexed `get` and `put` operations.

> **Project status:** design and build scaffolding only. No media implementation is present yet. The interfaces below describe the intended API and may change before the first release.

## Goals

- Return exactly the requested video frame, including across codecs that use bidirectionally predicted frames.
- Keep sequential access fast by retaining decoded frames and codec state between requests.
- Keep audio aligned to the video timeline for indexed reads, writes, seeking, and playback.
- Move frames through CPU buffers or GL/WebGL textures and framebuffers, using zero-copy paths when the platform permits.
- Read and write ISO Base Media File Format/MP4, initially targeting HEVC/H.265 and AV1 video with AAC audio.
- Present equivalent Rust and JavaScript concepts on native and `wasm32` targets.
- Make containers, codecs, storage, and graphics backends replaceable without exposing backend details in the common API.

The project is a library, not a media CLI or an FFmpeg binding. FFmpeg is a feature-set reference only; zvidlib will not copy or incorporate FFmpeg source.

## Planned API shape

The simplest file workflow is intended to look like this (illustrative API, not implemented):

```rust,ignore
let mut input = zvidlib::open("clip.mp4").await?;
let frame = input.video(0)?.get(5).await?;
let audio = input.audio(0)?.get(5).await?;

canvas.upload(&frame).await?;
```

Writing uses the same frame-indexed model:

```rust,ignore
let mut output = zvidlib::create("recording.mp4", options).await?;
output.video(0)?.put(5, canvas.capture()).await?;
output.audio(0)?.put(5, audio_window).await?;
output.finish().await?;
```

JavaScript wrappers will preserve `open`/`create`, stream selection, and `get(n)`/`put(n, value)` while returning Promises where browser APIs require asynchronous work.

Frame indices are zero-based. A video `get(n)` returns exactly frame `n` in presentation order, not merely the nearest keyframe. The matching audio `get(n)` returns the half-open sample interval covered by video frame `n`; sample-accurate APIs will also be available for audio-only use.

## Data paths

Video frames will support two families of destination/source:

- CPU-backed planes with explicit pixel format, dimensions, stride, color space, and transfer metadata.
- GPU-backed images associated with a GL texture or framebuffer on native targets and a WebGL texture or framebuffer in browsers.

GPU interop is capability-based. A backend may share an image without a copy, copy on the GPU, or fall back through CPU memory. Callers can require a particular behavior and receive an unsupported-capability error instead of an implicit expensive fallback.

Audio buffers carry their sample format, channel layout, sample rate, exact timeline interval, and any priming or padding information needed for gapless synchronization.

## Platform expectations

| Capability | Native | WebAssembly/browser |
| --- | --- | --- |
| Input/output | Files, memory, caller storage | `Blob`, streams, memory, File System Access handles when supplied |
| Video acceleration | Pluggable software or platform backend | WebCodecs when available, otherwise a compatible WASM backend |
| Graphics | OpenGL-family context supplied by caller | WebGL context supplied by caller |
| Audio | Raw buffers and pluggable device integration | Web Audio buffers/nodes supplied by caller |
| Concurrency | Worker threads where safe | Async tasks; workers/threads only when browser isolation permits |

Codec and hardware availability varies by browser and operating system. Opening media will report capabilities explicitly; container support does not imply that every platform can encode or decode every advertised codec.

## Repository layout

The repository currently contains documentation and a deliberately empty Rust library target so Cargo can validate native and WASM build configuration. Planned modules and dependency boundaries are described in [ARCHITECTURE.md](ARCHITECTURE.md). Runtime dependencies will be added only with a documented portability, maintenance, size, and licensing rationale.

## Building the scaffold

Install stable Rust with the `wasm32-unknown-unknown` target, then run:

```console
cargo check --features native
cargo check --target wasm32-unknown-unknown --no-default-features --features web
cargo fmt --all -- --check
cargo clippy --all-targets --features native -- -D warnings
```

These commands currently validate project setup only; they do not produce a functional media library.

## Roadmap

1. Specify and test timeline arithmetic, frame/audio value types, errors, and capability discovery.
2. Build incremental MP4 parsing and writing with deterministic fixture tests.
3. Add decoder/encoder traits and one end-to-end decode path before expanding codec coverage.
4. Add native GL and WebGL transfer backends, then synchronized playback and recording adapters.
5. Stabilize Rust and JavaScript APIs after cross-platform conformance testing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, seeking and caching behavior, MP4 responsibilities, and portability constraints.
