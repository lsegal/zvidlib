# Changelog

All notable changes to zvidlib will be documented in this file.

## Unreleased

- Add a dependency-free native lossless monochrome AV1 Main-profile encoder with standardized OBU access units and `av1C`, exact configurable timescale/frame-duration metadata, strict format/rate/resource validation, and independent FFmpeg round-trip conformance coverage.
- Add backend-independent HEVC/AV1 codec conformance infrastructure with canonical SHA-256 frame fingerprints, exact-frame decoder access-pattern checks, encoder round-trip quality validation, and a complete 120-frame expected-output vector for the bundled HEVC sample.
- Make the native GL example (`examples/native_gl/`) draw to a real cross-platform `winit`/`glutin` window instead of writing uploaded frames out as PPM files, with looping playback (matching the web example) and an on-screen FPS counter in the top-left corner.
- Fix the `web_canvas` example's canvas being a fixed 1280x720 size instead of stretching to fill its container (including on windows wider than the video's intrinsic 1280x720 resolution), and add an on-screen FPS counter in the top-left corner.
- Fix the browser `WebCodecs` video decoder (`WebVideoDecodeSession::get()`) getting stuck on the first ~12 frames of content with sparse key frames (e.g. the bundled `examples/media/BigBuckBunny.mp4` sample, a single key frame followed by ~120 delta frames): it now keeps a decode session open across `get()` calls instead of resetting and flushing on every call, removing the artificial 12-frame batch cap so `get()` is bounded only by `Limits::max_decode_samples_per_seek`.
- Add `examples/web_canvas/package.json` with a `pnpm dev` script that builds the WebAssembly package straight into `examples/web_canvas/pkg` and starts a Vite dev server, replacing the previous manual `wasm-pack build` + `cp -r` + `python -m http.server` steps.
- Add a real compressed video decoder backend for the browser (`web`) build, backed by the browser's native `WebCodecs` `VideoDecoder`, so `VideoStream.get()` decodes actual HEVC/AV1 MP4 samples into RGBA frames instead of always rejecting with `UNSUPPORTED`; the `web_canvas` example now renders real decoded frames when the browser supports the sample's codec.
- Add an `examples/` directory with runnable native GL and browser WebGL canvas examples driven against a Big Buck Bunny MP4 sample checked into the repo at `examples/media/BigBuckBunny.mp4` (shared with the web canvas example via a symlink), so no manual download is required.
- Fix the seekable MP4 muxer emitting an `stsc` entry for tracks finished without any samples, which referenced a chunk that did not exist.
- Document the required non-seekable `ByteSink` contract and add tests verifying a sink that overrides `is_seekable()` rejects `seek()` with `ErrorKind::Unsupported` and that ordinary MP4 creation rejects such a sink before writing any bytes.
- Publish API documentation to GitHub Pages and update the README to describe the current pre-1.0 status instead of "in development".
- Add exact AAC sample-range reads with gapless/edit mapping and audio-clock-driven native/Web Audio playback synchronization with cancellable seek preroll.
- Add bounded incremental MP4 probing and read-only ordinary/fragmented sample indexes with decode and presentation timing, byte ranges, dependencies, edits, and codec configuration.
- Add the browser WebAssembly package with BigInt-safe timeline values, stable JavaScript errors, cancellable Blob and stream input, owned typed-array media values, stream and playback handles, and Blob output.
- Add normalized video codec factories and a bounded exact-frame decoder path with presentation-order reordering, cancellation, and a portable uncompressed conformance backend.
- Add strict synchronized indexed video/audio writing with portable encoder contracts, bounded sink backpressure, gapless finalization, and deterministic seekable MP4 muxing.
- Add explicit CPU, native OpenGL, and browser WebGL frame transfer contracts with inspectable copy modes, conversion stages, context validation, strict fallback policies, and caller-safe resource ownership.
- Add the first portable implementation with checked timeline arithmetic, synchronized audio intervals, validated media buffers, capability and error types, asynchronous memory I/O, and native/WASM tests.
- Document complete planned native GL and browser WebGL workflows for synchronized video/audio reading and writing, including WebAssembly packaging and use.
- Initialize the project with its native and WebAssembly architecture, public API design, and Rust build scaffolding.
