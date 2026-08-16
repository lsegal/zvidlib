# Changelog

All notable changes to zvidlib will be documented in this file.

## Unreleased

- Add exact AAC sample-range reads with gapless/edit mapping and audio-clock-driven native/Web Audio playback synchronization with cancellable seek preroll.
- Add bounded incremental MP4 probing and read-only ordinary/fragmented sample indexes with decode and presentation timing, byte ranges, dependencies, edits, and codec configuration.
- Add the browser WebAssembly package with BigInt-safe timeline values, stable JavaScript errors, cancellable Blob and stream input, owned typed-array media values, stream and playback handles, and Blob output.
- Add normalized video codec factories and a bounded exact-frame decoder path with presentation-order reordering, cancellation, and a portable uncompressed conformance backend.
- Add strict synchronized indexed video/audio writing with portable encoder contracts, bounded sink backpressure, gapless finalization, and deterministic seekable MP4 muxing.
- Add explicit CPU, native OpenGL, and browser WebGL frame transfer contracts with inspectable copy modes, conversion stages, context validation, strict fallback policies, and caller-safe resource ownership.
- Add the first portable implementation with checked timeline arithmetic, synchronized audio intervals, validated media buffers, capability and error types, asynchronous memory I/O, and native/WASM tests.
- Document complete planned native GL and browser WebGL workflows for synchronized video/audio reading and writing, including WebAssembly packaging and use.
- Initialize the project with its native and WebAssembly architecture, public API design, and Rust build scaffolding.
