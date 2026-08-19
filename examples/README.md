# Examples

These examples demonstrate zvidlib's currently implemented API against a real MP4 sample: a short,
re-encoded clip of the CC-BY-licensed [Big Buck Bunny](https://peach.blender.org/) film by the
Blender Foundation. The clip is checked into the repo at `examples/media/BigBuckBunny.mp4`
(trimmed to a few seconds and re-encoded small so it's cheap to keep in git), so no manual
download is needed before running either example. `examples/web_canvas/BigBuckBunny.mp4` is a
symlink to that same file so both examples share one copy.

## Compressed decoding

As documented in the main [README](../README.md#implemented-browser-boundary), the browser
(`web`) build now decodes the sample's real HEVC video track through the browser's native
`WebCodecs` `VideoDecoder`, so `web_canvas/` renders real decoded pixels instead of a synthetic
gradient. Whether that decode actually succeeds depends on the browser and platform: zvidlib
queries `VideoDecoder.isConfigSupported()` first, and `video.get()` still rejects with
`UNSUPPORTED` (falling back to the synthetic gradient) if the browser has no HEVC decoder
available. The native example has no `WebCodecs` equivalent, so it still only demuxes the real
sample's track metadata and does not decode pixels.

## Native GL: `native_gl.rs`

Opens the sample, prints its demuxed video track metadata, and drives the real CPU → native
OpenGL upload path (`execute_transfer`) through a minimal in-process `GraphicsAdapter` that stands
in for a real GL context, so the example runs without a window or GL dependencies. Each uploaded
"texture" is written out as a PPM image under `examples/output/native_gl/` so the transfer is
observable.

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
