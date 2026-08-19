# Examples

These examples demonstrate zvidlib's currently implemented API against a real MP4 sample: a short,
re-encoded clip of the CC-BY-licensed [Big Buck Bunny](https://peach.blender.org/) film by the
Blender Foundation. The clip is checked into the repo at `examples/media/BigBuckBunny.mp4`
(trimmed to a few seconds and re-encoded small so it's cheap to keep in git), so no manual
download is needed before running either example. `examples/web_canvas/BigBuckBunny.mp4` is a
symlink to that same file so both examples share one copy.

## Current limitation: no compressed decoder backend yet

As documented in the main [README](../README.md#implemented-browser-boundary), zvidlib has not
yet registered a compressed (HEVC/AV1) video decoder backend, so indexed video `get`/decode calls
reject with `UNSUPPORTED`. Both examples still demux the real sample file to read its actual track
metadata (dimensions, codec, sample count, timing), but they render synthetic frames sized to that
track instead of real decoded pixels. Once a decoder backend lands, replace the synthetic frame
generator with `ExactFrameReader::get`/`Playback` output; everything downstream (the transfer
contract, canvas upload) is unchanged.

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
same CPU → WebGL upload path via the generated `zvidlib` package. It renders synthetic frames the
same way as the native example and shows the real `UNSUPPORTED` error zvidlib returns for
`video.get()` until a decoder backend is registered.

Build the WebAssembly package first (see the main [README](../README.md#building-testing-and-using-webassembly)):

```console
wasm-pack build --target web --out-dir pkg --no-default-features --features web
```

Then copy or symlink `pkg/` next to `examples/web_canvas/index.html` and serve the directory over
HTTP(S) (`BigBuckBunny.mp4` is already symlinked in):

```console
cp -r pkg examples/web_canvas/pkg
python -m http.server 8000 --directory examples/web_canvas
```

Open `http://localhost:8000/` and click **Play**.
