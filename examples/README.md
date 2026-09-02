# Examples

These examples demonstrate zvidlib's currently implemented API against a real MP4 sample: a
1080p, 32-second, re-encoded clip of the CC-BY-licensed [Big Buck Bunny](https://peach.blender.org/)
film by the Blender Foundation. The clip is checked into the repo at
`examples/media/BigBuckBunny.mp4` (re-encoded small so it's cheap to keep in git), so no manual
download is needed before running either example. `examples/web_canvas/BigBuckBunny.mp4` is a
symlink to that same file so both examples share one copy.

## Compressed decoding

As documented in the main [README](../README.md#implemented-browser-boundary), the browser
(`web`) build now decodes the sample's real HEVC video track through the browser's native
`WebCodecs` `VideoDecoder`, so `web_canvas/` renders real decoded pixels instead of a synthetic
gradient. Whether that decode actually succeeds depends on the browser and platform: zvidlib
queries `VideoDecoder.isConfigSupported()` first, and `video.get()` still rejects with
`UNSUPPORTED` (falling back to the synthetic gradient) if the browser has no HEVC decoder
available. The native build prefers NVIDIA NVDEC on supported 64-bit Windows/Linux systems, then
the D3D11-aware Media Foundation HEVC decoder on Windows, or VideoToolbox on macOS. It falls back
to zvidlib's dependency-free pure-Rust HEVC Main decoder when acceleration is unavailable, so the
native OpenGL example still renders the same decoded pixels without requiring a system codec.

## Native GL: `native_gl/`

Opens the sample, selects accelerated HEVC Main decoding when available (printing the selected
`CodecImplementation`), extracts the AAC-LC access units from the same MP4 sample index, and plays
decoded PCM through the default native audio device. The `PlaybackController` uses the audio
device's sample clock to choose exact video presentation frames from the MP4 PTS/duration map, then
uploads those decoded RGBA pixels through the real CPU → native OpenGL path (`execute_transfer`).
Playback loops back to the first frame after end-of-stream, matching the web example's looping
behavior, and the decoded-frame playback rate is drawn in the window's top-left corner.

Playback controls (also printed at startup): `SPACE` toggles play/pause, `LEFT`/`RIGHT` step to the
previous/next frame, `J`/`L` rewind/fast-forward five seconds, and the timeline bar drawn along the
bottom edge scrubs when you click or drag it. Hovering only moves the bar's marker. Seeking keeps
the audio clock and the displayed video frame in sync, and audio keeps playing while scrubbing if
playback was running.

A drag previews on a background thread rather than seeking: a second decoder answers the newest
pointer position, snapped back to its random-access point so each preview is one intra picture, and
supersedes any position the pointer has already moved past. The window keeps drawing throughout,
and only the frame the pointer is released on is seeked to and decoded exactly.

```console
cargo run --example native_gl --features native
```

## Web canvas: `web_canvas/`

A browser page that opens the sample as a `Blob`, creates a WebGL2 canvas context, extracts the
indexed AAC-LC access units from zvidlib's MP4 demuxer, and schedules decoded PCM through Web
Audio with WebCodecs `AudioDecoder` after the play button's user gesture. It calls `video.get()`
for each displayed frame and uploads the real decoded RGBA pixels; if the browser cannot decode
HEVC it falls back to a synthetic gradient sized to the real track instead. Both paths use MP4
sample timing and, when audio is available, the `AudioContext` clock rather than display-refresh
or hard-coded FPS pacing.

The page's controls are play/pause, five-second rewind/fast-forward, previous/next frame stepping,
and a timeline range input that scrubs when you click or drag it, keeping only the newest requested
position when a drag outruns the decoder. A drag draws every frame it passes rather than only the
one it lands on: forwards it steps a frame at a time, and backwards it restarts at the random-access
point at or before the pointer (`VideoStream.randomAccessPoints()`) and walks forwards from there,
so the picture follows the pointer in both directions. The whole AAC track is
decoded once into a single `AudioBuffer`, so seeking and scrubbing only reschedule playback from
the new offset instead of re-running the decoder, and audio keeps playing while you scrub.

`examples/web_canvas/package.json` wraps the whole setup in one command. From `examples/web_canvas/`:

```console
pnpm install
pnpm dev
```

`pnpm dev` builds the WebAssembly package straight into `examples/web_canvas/pkg` with
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) (install it first if you don't have
it: `cargo install wasm-pack`), then starts a [Vite](https://vitejs.dev/) dev server for the
directory. Open the printed `http://localhost:5173/` URL and click **Play**.

Requires `wasm-pack` on `PATH` and [`pnpm`](https://pnpm.io/installation). See the main
[README](../README.md#building-testing-and-using-webassembly) for the equivalent manual
`wasm-pack build` command if you'd rather run it outside this workflow.
