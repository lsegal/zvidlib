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
available. On 64-bit Windows, the native build prefers NVIDIA NVDEC and then the D3D11-aware Media
Foundation HEVC decoder when the installed driver/codec supports the requested stream. It falls
back to zvidlib's dependency-free pure-Rust HEVC Main decoder everywhere else, so the native
OpenGL example still renders the same decoded pixels without requiring a system codec.

## Native GL: `native_gl/`

Opens the sample, selects accelerated HEVC Main decoding when available (printing the selected
`CodecImplementation`), and drives the real CPU → native OpenGL upload path (`execute_transfer`)
through a `GraphicsAdapter` backed by a real `winit` window and `glutin` OpenGL context on Windows,
macOS, and Linux. Playback loops back to the first frame once the last one has been shown, matching
the web example's looping behavior, and the decoded-frame playback rate is drawn in the window's
top-left corner.

```console
cargo run --example native_gl --features native
```

## Web canvas: `web_canvas/`

A browser page that opens the sample as a `Blob`, creates a WebGL2 canvas context, and drives the
same CPU → WebGL upload path via the generated `zvidlib` package. It calls `video.get()` for each
displayed frame and uploads the real decoded RGBA pixels; if the browser cannot decode HEVC it
falls back to a synthetic gradient sized to the real track instead.

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
