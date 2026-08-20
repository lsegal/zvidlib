# zvidlib

zvidlib is a Rust library for frame-accurate video and synchronized audio I/O on native and WebAssembly targets. Its primary jobs are reading an MP4 into a GL/WebGL canvas and writing canvas frames plus an audio stream into an MP4, behind a small API centered on indexed `get` and `put` operations.

> **Project status:** zvidlib is pre-1.0 and its API may still change before the first release. The portable foundation now includes checked timeline arithmetic, validated media buffers, asynchronous byte I/O, CPU/GL/WebGL transfer contracts, bounded ordinary/fragmented MP4 sample indexing, normalized codec factories, bounded exact-frame video decoding, exact AAC sample reads, audio-clock playback control and adapter contracts, encoder contracts, strict indexed output, seekable MP4 muxing, and the browser WebAssembly boundary. The generated JavaScript package includes BigInt-safe values, stable errors, Blob/stream input, Blob output, and session/stream/playback handles, and the browser (`web`) build decodes real HEVC/AV1 video through the browser's native `WebCodecs` `VideoDecoder`. Compressed video encoding and concrete audio-device and `AudioContext` bindings remain planned, so those backend-dependent JavaScript operations currently reject with `UNSUPPORTED`. The complete workflow interfaces below remain intentionally aspirational and may change before the first release.

## Documentation

Full API documentation is published at [lsegal.github.io/zvidlib](https://lsegal.github.io/zvidlib/) and rebuilt from `main` on every push.

## Goals

- Return exactly the requested video frame, including across codecs that use bidirectionally predicted frames.
- Keep sequential access fast by retaining decoded frames and codec state between requests.
- Keep audio aligned to the video timeline for indexed reads, writes, seeking, and playback.
- Move frames through CPU buffers or GL/WebGL textures and framebuffers, using zero-copy paths when the platform permits.
- Read and write ISO Base Media File Format/MP4, initially targeting HEVC/H.265 and AV1 video with AAC audio.
- Present equivalent Rust and JavaScript concepts on native and `wasm32` targets.
- Make containers, codecs, storage, and graphics backends replaceable without exposing backend details in the common API.

The project is a library, not a media CLI or an FFmpeg binding. FFmpeg is a feature-set reference only; zvidlib will not copy or incorporate FFmpeg source.

## Implemented browser boundary

The `web` feature exposes `MediaInput`, `MediaOutput`, `Playback`, `VideoStream`, `AudioStream`, `OpenOptions`, `CreateOptions`, `PlaybackOptions`, `FrameIndex`, `Timestamp`, `Rational`, `SampleRange`, `VideoFrame`, and `AudioBuffer` through `wasm-bindgen`.

`MediaInput.open` accepts a `Blob`, `ReadableStream<Uint8Array>`, `ArrayBuffer`, or typed-array view. It consumes streams, always releases its reader lock, and supports cancellation through `OpenOptions.signal`. Input bytes are copied into owned WebAssembly storage; `bytes()` returns a fresh JavaScript snapshot rather than a view into growable WebAssembly memory.

```js
import init, { FrameIndex, MediaInput, OpenOptions, errorCode } from "./pkg/zvidlib.js";

await init();

const controller = new AbortController();
const options = new OpenOptions(64n * 1024n * 1024n);
options.signal = controller.signal;

const response = await fetch("./clip.mp4");
const input = await MediaInput.open(await response.blob(), options);
console.log(input.byteLength); // BigInt
console.log(new FrameIndex(18_446_744_073_709_551_615n).value);

try {
  // Decodes the real frame via the browser's native WebCodecs `VideoDecoder`
  // for HEVC/AV1 input tracks.
  const frame = await input.video(0).get(0n);
  console.log(frame.width, frame.height, frame.pixels.length);
} catch (error) {
  // "UNSUPPORTED" if this browser/platform has no decoder for the track's
  // codec, rather than a fake or nearest frame.
  console.log(errorCode(error));
}

input.close();
```

`MediaOutput.finish()` returns a browser-owned `Blob` with the configured MIME type. `writeEncodedChunk()` is the byte-sink boundary used by a muxer backend; indexed video/audio `put` calls remain explicitly unsupported until those backends land.

```js
import { CreateOptions, MediaOutput } from "./pkg/zvidlib.js";

const createOptions = new CreateOptions("mp4");
createOptions.maxOutputBytes = 512n * 1024n * 1024n;
const output = await MediaOutput.create(createOptions);

// A future MP4 muxer supplies already-encoded container chunks here.
output.writeEncodedChunk(new Uint8Array([0, 0, 0, 8, 0x66, 0x72, 0x65, 0x65]));
const blob = await output.finish();
console.log(blob.type); // "video/mp4"
```

All exported 64-bit frame, sample, and timestamp values return JavaScript `BigInt`. Inputs accept `BigInt` across the full Rust range or validated `Number` values only within JavaScript's safe-integer range. Rust and boundary failures reject with native `Error` instances named `ZvidError`; their stable `code` values can be read directly or with `errorCode(error)`.

## Planned API examples

The examples in this section show the intended complete workflows. Rust examples are deliberately marked `ignore`, and browser examples depend on media backends that are not fully registered yet: video decoding works (see [Implemented browser boundary](#implemented-browser-boundary)), but `Playback`, audio decode/encode, and video encode still reject with `UNSUPPORTED` until their implementations land. The boundary classes exist for all of these. Names may change before the first release, while the ownership, synchronization, and cleanup steps are part of the design.

### Read, play audio, and render with native OpenGL

The application owns the window, current GL context, and audio device. zvidlib owns the demuxer and decoder state, uploads exact frames into the caller's texture, and uses the audio device clock as the playback clock.

```rust,ignore
use std::time::Duration;
use zvidlib::{AudioOutput, GlContext, OpenOptions, Playback};

#[tokio::main]
async fn main() -> zvidlib::Result<()> {
    // Application-specific setup (glutin/winit, SDL, etc.). The GL context must
    // stay current on this thread for every zvidlib and draw call below.
    let (window, gl): (_, GlContext) = create_window_and_current_gl_context()?;
    let audio_device: AudioOutput = open_default_audio_device()?;
    let texture = create_rgba_texture(&gl, window.drawable_size())?;

    let input = zvidlib::open("clip.mp4", OpenOptions::default()).await?;
    let video = input.video(0)?;
    let audio = input.audio(0)?;
    let mut playback = Playback::builder(video, audio)
        .audio_output(audio_device)
        .video_destination(gl.texture_2d(texture))
        .build()
        .await?;

    playback.play().await?;
    while !window.should_close() && !playback.is_finished() {
        window.poll_events();

        // `present` schedules audio ahead, asks `get(n)` for the exact frame
        // selected by the audio clock, and uploads that frame to `texture`.
        if playback.present().await? {
            draw_fullscreen_texture(&gl, texture)?;
            window.swap_buffers()?;
        }

        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    playback.stop().await?;
    input.close().await?;
    Ok(())
}
```

The helpers that create the native window, GL objects, shaders, and audio device depend on the application's windowing/audio crates and are intentionally left to the host. `present` never substitutes a nearby keyframe: it resolves the current presentation index and calls the same exact-frame path as `video.get(n)`.

### Read, play audio, and render with WebGL

Build the package as described in [Building and using WebAssembly](#building-and-using-webassembly), then serve this page and `clip.mp4` from the same directory. Browser security rules require HTTP(S); opening the HTML as a `file:` URL will not work reliably.

```html
<!doctype html>
<html lang="en">
  <meta charset="utf-8">
  <title>zvidlib WebGL playback</title>
  <canvas id="video" width="1280" height="720"></canvas>
  <button id="play" type="button">Play</button>
  <script type="module">
    import init, { MediaInput, Playback } from "./pkg/zvidlib.js";

    await init();

    const canvas = document.querySelector("#video");
    const playButton = document.querySelector("#play");
    const gl = canvas.getContext("webgl2", { alpha: false });
    if (!gl) throw new Error("WebGL 2 is unavailable");

    const response = await fetch("./clip.mp4");
    if (!response.ok) throw new Error(`clip.mp4: ${response.status}`);

    const input = await MediaInput.open(await response.blob());
    const video = input.video(0);
    const audio = input.audio(0);
    const audioContext = new AudioContext();
    const playback = await Playback.create({
      video,
      audio,
      audioContext,
      webgl: { context: gl, canvas },
    });

    let animationFrame = 0;
    async function render() {
      // Schedules audio and renders the exact `video.get(n)` result selected by
      // the AudioContext clock into the canvas's default framebuffer.
      const { finished } = await playback.present();
      if (!finished) animationFrame = requestAnimationFrame(render);
    }

    playButton.addEventListener("click", async () => {
      playButton.disabled = true;
      await audioContext.resume(); // Must follow a user gesture.
      await playback.play();
      animationFrame = requestAnimationFrame(render);
    }, { once: true });

    window.addEventListener("pagehide", () => {
      cancelAnimationFrame(animationFrame);
      playback.close();
      input.close();
      audioContext.close();
    }, { once: true });
  </script>
</html>
```

### Write synchronized native GL and audio input

Writing uses the same zero-based frame index. The application captures audio for the exact half-open interval represented by each video frame and submits both values before advancing the index.

```rust,ignore
use zvidlib::{CreateOptions, GlContext, VideoCodec};

#[tokio::main]
async fn main() -> zvidlib::Result<()> {
    let (window, gl): (_, GlContext) = create_window_and_current_gl_context()?;
    let microphone = open_default_audio_input()?;
    let framebuffer = create_recording_framebuffer(&gl, 1920, 1080)?;

    let mut output = zvidlib::create(
        "recording.mp4",
        CreateOptions::mp4()
            .video(VideoCodec::Av1, 1920, 1080, 30, 1)
            .aac_audio(microphone.sample_rate(), microphone.channels()),
    ).await?;
    let video = output.video(0)?;
    let audio = output.audio(0)?;

    for frame_index in 0_u64..300 {
        render_scene_into(&gl, framebuffer, frame_index)?;
        let samples = microphone.capture(audio.interval_for_frame(frame_index)?).await?;

        video.put(frame_index, gl.framebuffer(framebuffer)).await?;
        audio.put(frame_index, samples).await?;
    }

    // `finish` drains delayed video/audio packets and writes the MP4 indexes.
    output.finish().await?;
    window.close();
    Ok(())
}
```

### Write a WebGL canvas and Web Audio stream

This browser example records ten seconds. A real application can replace `drawScene` and `microphone` with any WebGL renderer and timestamped Web Audio source.

```js
import init, { MediaOutput } from "./pkg/zvidlib.js";

await init();

const canvas = document.querySelector("#video");
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
const audioContext = new AudioContext();
const microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
const output = await MediaOutput.create({
  container: "mp4",
  video: { codec: "av1", width: canvas.width, height: canvas.height, fps: 30 },
  audio: { codec: "aac", sampleRate: audioContext.sampleRate, channels: 2 },
});
const video = output.video(0);
const audio = output.audio(0);
const capture = await audio.captureFromMediaStream(audioContext, microphone);

for (let frameIndex = 0n; frameIndex < 300n; frameIndex++) {
  drawScene(gl, frameIndex);
  const samples = await capture.get(audio.intervalForFrame(frameIndex));
  await video.put(frameIndex, { webgl: gl, framebuffer: null });
  await audio.put(frameIndex, samples);
}

capture.close();
microphone.getTracks().forEach((track) => track.stop());
const mp4 = await output.finish();
const link = Object.assign(document.createElement("a"), {
  href: URL.createObjectURL(mp4),
  download: "recording.mp4",
  textContent: "Download recording",
});
document.body.append(link);
```

JavaScript frame indices are `BigInt` so the wrapper can preserve the Rust `u64` range. Both writers reject skipped or repeated indices by default, and `finish()` is required to drain codecs and finalize the MP4.

The implemented portable writer core exposes `VideoEncoder` and `AudioEncoder` contracts plus `MediaOutput`. An encoder backend declares its MP4 codec configuration and exact output timescale, then returns `EncodedSample` values with DTS, PTS, duration, sync, and dependency metadata. `MediaOutput::put_video` accepts the shared CPU/GL/WebGL `FrameSource`; it and `put_audio` enforce zero-based consecutive indices and exact frame-aligned audio ranges. `finish` drains both encoders, records audio priming and padding in an edit list, finalizes sample indexes, and flushes the seekable `ByteSink`.

Frame indices are zero-based. A video `get(n)` returns exactly frame `n` in presentation order, not merely the nearest keyframe. The matching audio `get(n)` returns the half-open sample interval covered by video frame `n`; sample-accurate APIs will also be available for audio-only use.

Runnable versions of the native GL and browser WebGL workflows above, driven against a real Big Buck Bunny MP4 sample, live in [`examples/`](examples/README.md).

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

The repository contains a dependency-free portable core that validates native and WASM build configuration and implements the foundational timeline, media, I/O, frame-transfer, bounded MP4 reading, codec factory, exact-frame decoding, encoder, indexed-output, and seekable MP4 writing layers. Planned modules and dependency boundaries are described in [ARCHITECTURE.md](ARCHITECTURE.md). Runtime dependencies will be added only with a documented portability, maintenance, size, and licensing rationale.

## Building the library

Install stable Rust with the `wasm32-unknown-unknown` target, then run:

```console
cargo check --features native
cargo check --target wasm32-unknown-unknown --no-default-features --features web
cargo fmt --all -- --check
cargo clippy --all-targets --features native -- -D warnings
```

These commands validate the portable core on both targets. No concrete video or audio codec backend is bundled yet, so they do not by themselves produce an end-to-end media reader or encoded recording.

Native compressed-codec backends use the public `VideoDecoderConformanceVector`
and `VideoEncoderConformanceVector` runners before registration. Decoder vectors
pin canonical SHA-256 fingerprints for every presentation frame and are tested
under sequential, reverse, and seek-heavy access. Encoder vectors validate
standard configuration and packet timing, then decode through an independently
conforming backend and enforce an explicit PSNR floor. This gives HEVC and AV1
work finite, reusable acceptance targets without delegating codec behavior to an
external library.

## Building, testing, and using WebAssembly

The `web` feature excludes native-only integrations. During scaffold development, install the Rust target and verify it directly:

```console
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --no-default-features --features web
```

Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) and create a browser-ready ES module package:

```console
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --no-default-features --features web
python -m http.server 8000
```

The build creates `pkg/zvidlib.js`, `pkg/zvidlib.d.ts`, `pkg/zvidlib_bg.wasm`, and package metadata. Open `http://localhost:8000/` and import the generated module with `import init, { ... } from "./pkg/zvidlib.js"`, as in the browser examples. Call `await init()` exactly once before using an exported class. Deploy the generated JavaScript and WASM files together, with the server returning `application/wasm` for the `.wasm` file.

For the browser example specifically, `examples/web_canvas/package.json` wraps the `wasm-pack build` and serving steps above into one `pnpm dev` command; see [examples/README.md](examples/README.md#web-canvas-web_canvas).

Run the browser integration suite in an installed Chrome browser with:

```console
wasm-pack test --headless --chrome --no-default-features --features web
```

The suite verifies Blob and stream input, cancellation, reader-lock cleanup, BigInt range handling, typed-array copy lifetimes, browser-object ownership, stable errors, Blob output, and decoding the bundled HEVC sample through WebCodecs. The base build does not require WASM threads or cross-origin isolation. Future optional threaded builds will document their additional headers and browser requirements separately.

The `web` feature's video decoder uses `web-sys`'s `WebCodecs` bindings, which that crate gates behind `--cfg=web_sys_unstable_apis` because the spec is still evolving. `.cargo/config.toml` sets that flag for the `wasm32-unknown-unknown` target automatically, so the `cargo`/`wasm-pack` commands above need no extra flags.

## Roadmap

1. Specify and test timeline arithmetic, frame/audio value types, errors, and capability discovery.
2. Build incremental MP4 parsing and writing with deterministic fixture tests.
3. Add decoder/encoder traits and one end-to-end decode path before expanding codec coverage.
4. Add native GL and WebGL transfer backends, then synchronized playback and recording adapters.
5. Stabilize Rust and JavaScript APIs after cross-platform conformance testing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, seeking and caching behavior, MP4 responsibilities, and portability constraints.
