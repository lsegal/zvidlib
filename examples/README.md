# Examples

These examples demonstrate zvidlib's currently implemented API against a real MP4 sample: a
1080p, 32-second, re-encoded clip of the CC-BY-licensed [Big Buck Bunny](https://peach.blender.org/)
film by the Blender Foundation. The clip is checked into the repo at
`examples/media/BigBuckBunny.mp4` (re-encoded small so it's cheap to keep in git), so no manual
download is needed before running either example. `examples/web_canvas/BigBuckBunny.mp4` is a
symlink to that same file so both examples share one copy.

The same clip is bundled a second time at `examples/media/BigBuckBunny.av1.mp4`, with an AV1 Main
track at 540p and the same AAC audio. Its video track is the only difference that matters: a stock
Chrome has no HEVC decoder, so the browser example could not decode the original sample at all and
fell back to a synthetic gradient — which took the seek preview tier down with it, since a preview
is a decoded picture like any other. `examples/web_canvas/samples.js` lists both, and the page asks
the browser which it can decode before fetching either, so the HEVC copy is still what a browser
that can decode it gets. The native example uses only the HEVC copy.

## Compressed decoding

As documented in the main [README](../README.md#implemented-browser-boundary), the browser
(`web`) build now decodes the sample's real video track through the browser's native `WebCodecs`
`VideoDecoder`, so `web_canvas/` renders real decoded pixels instead of a synthetic gradient.
Which of the two bundled samples it decodes depends on the browser and platform: the page probes
`VideoDecoder.isConfigSupported()` for each declared codec string before fetching anything, opens
the first one that both reports support and decodes frame zero, and only draws the synthetic
gradient when neither an HEVC nor an AV1 decoder is available. The native build prefers NVIDIA NVDEC on supported 64-bit Windows/Linux systems, then
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

A drag previews on a background thread rather than seeking: the worker walks towards the newest
pointer position and supersedes any position the pointer has already moved past. It publishes a
picture roughly every 150 ms of decoding rather than every frame it passes, so the picture keeps
moving during a long walk without paying a full-resolution conversion per frame, and a position
behind the reader restarts at its random-access point. The window keeps drawing throughout, and
only the frame the pointer is released on is seeked to and decoded exactly.

What the drag draws *first* does not come from that walk at all. A background pass over the track
keeps a quarter-scale picture every half second of playback, capped at 64 MiB, and a drag looks the
nearest one up and uploads it before it asks the walk for anything - a lookup and a texture upload,
under a millisecond, against the second and a half the same position costs to decode exactly on the
bundled sample, whose 768 frames are a single group of pictures. The walk still runs underneath and
replaces the preview with the exact frame when it lands. Until the pass reaches a point, a drag
there falls back to the nearest earlier picture, so the bar is progressively right rather than
blank.

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
position when a drag outruns the decoder. A drag draws pictures on the way to the frame it lands on
rather than only that frame: it walks forwards from where the decoder already is, or, when the
pointer moves backwards, from the random-access point at or before it
(`VideoStream.randomAccessPoints()`). Each step covers whatever fits in 150 ms at the rate the walk
is decoding at, so the picture keeps moving throughout a drag without drawing - and paying for -
every frame it passes.

The picture the drag *follows*, though, is not the walk's. `ARCHITECTURE.md` section 3.2 requires a
seek to any position of any track to answer in under 50 ms, and no walk can: the bundled sample
codes its 768 frames as one group of pictures, so the far end of the bar is 767 reference decodes
from the only place a decode can start. So the page builds `VideoStream.previews()` - the browser's
seek preview tier, one shrunk picture every stride frames - and every pointer sample draws the
nearest of them immediately while the walk goes after the exact frame underneath it. The pass has no
thread to fill itself on in a browser, so the page advances it one preview per `requestIdleCallback`
and it yields to the event loop in between; a lookup is answered from whatever the pass has reached
so far, so a drag over the far end works before the pass gets there. The AV1 copy of the sample is
coded the same way — 768 frames, one sync sample — so the tier is what answers a drag there too,
rather than the choice of sample quietly changing what the example demonstrates. The overlay under the frame
rate reports what each seek cost against `seekLatencyBudgetMs()`. The whole AAC track is
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
