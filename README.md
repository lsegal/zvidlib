# zvidlib

zvidlib is a Rust library under active development for frame-accurate video and synchronized audio I/O on native and WebAssembly targets. Its primary jobs are reading an MP4 into a GL/WebGL canvas and writing canvas frames plus an audio stream into an MP4, behind a small API centered on indexed `get` and `put` operations.

> **Project status:** the portable foundation now includes checked rational timeline arithmetic, synchronized frame/audio intervals, validated CPU media buffers, capability and error types, asynchronous byte I/O, encoder contracts, strict indexed output, and seekable MP4 muxing. Codec backends, MP4 reading, graphics transfer, playback adapters, and JavaScript APIs remain planned. The interfaces below describe the intended complete API and may change before the first release.

## Goals

- Return exactly the requested video frame, including across codecs that use bidirectionally predicted frames.
- Keep sequential access fast by retaining decoded frames and codec state between requests.
- Keep audio aligned to the video timeline for indexed reads, writes, seeking, and playback.
- Move frames through CPU buffers or GL/WebGL textures and framebuffers, using zero-copy paths when the platform permits.
- Read and write ISO Base Media File Format/MP4, initially targeting HEVC/H.265 and AV1 video with AAC audio.
- Present equivalent Rust and JavaScript concepts on native and `wasm32` targets.
- Make containers, codecs, storage, and graphics backends replaceable without exposing backend details in the common API.

The project is a library, not a media CLI or an FFmpeg binding. FFmpeg is a feature-set reference only; zvidlib will not copy or incorporate FFmpeg source.

## Planned API examples

The examples in this section show the intended complete workflows. They are deliberately marked `ignore` because the repository is still a scaffold and does not yet export these APIs. The names may change before the first release, but the ownership, synchronization, and cleanup steps are part of the design.

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

The implemented portable writer core exposes `VideoEncoder` and `AudioEncoder` contracts plus `MediaOutput`. An encoder backend declares its MP4 codec configuration and exact output timescale, then returns `EncodedSample` values with DTS, PTS, duration, sync, and dependency metadata. `MediaOutput::put_video` and `put_audio` enforce zero-based consecutive indices and exact frame-aligned audio ranges. `finish` drains both encoders, records audio priming and padding in an edit list, finalizes sample indexes, and flushes the seekable `ByteSink`. The associated video frame type is backend-defined so CPU, GL, and WebGL transfer implementations can use the same output state machine.

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

The repository contains a dependency-free portable core that validates native and WASM build configuration and implements the foundational values, I/O, encoder, indexed-output, and seekable MP4 writing layers. Planned modules and dependency boundaries are described in [ARCHITECTURE.md](ARCHITECTURE.md). Runtime dependencies will be added only with a documented portability, maintenance, size, and licensing rationale.

## Building the library

Install stable Rust with the `wasm32-unknown-unknown` target, then run:

```console
cargo check --features native
cargo check --target wasm32-unknown-unknown --no-default-features --features web
cargo fmt --all -- --check
cargo clippy --all-targets --features native -- -D warnings
```

These commands validate the portable core on both targets. No concrete video or audio codec backend is bundled yet, so they do not by themselves produce an end-to-end media reader or encoded recording.

## Building and using WebAssembly

The `web` feature excludes native-only integrations. During scaffold development, install the Rust target and verify it directly:

```console
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --no-default-features --features web
```

Once the JavaScript wrapper described above lands, install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) and create a browser-ready package:

```console
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --no-default-features --features web
python -m http.server 8000
```

Open `http://localhost:8000/` and import the generated module with `import init, { ... } from "./pkg/zvidlib.js"`, as in the browser examples. Call `await init()` exactly once before using an exported class. Deploy the generated `pkg/zvidlib.js` and `pkg/zvidlib_bg.wasm` together, with the server returning `application/wasm` for the `.wasm` file.

`wasm-pack build` currently packages the portable core only: `MediaInput`, `MediaOutput`, and `Playback` will become available as the implementation milestones land. The base build does not require WASM threads or cross-origin isolation. Future optional threaded builds will document their additional headers and browser requirements separately.

## Roadmap

1. Specify and test timeline arithmetic, frame/audio value types, errors, and capability discovery.
2. Build incremental MP4 parsing and writing with deterministic fixture tests.
3. Add decoder/encoder traits and one end-to-end decode path before expanding codec coverage.
4. Add native GL and WebGL transfer backends, then synchronized playback and recording adapters.
5. Stabilize Rust and JavaScript APIs after cross-platform conformance testing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, seeking and caching behavior, MP4 responsibilities, and portability constraints.
