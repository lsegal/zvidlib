# Changelog

All notable changes to zvidlib will be documented in this file.

## Unreleased

- Add strict synchronized indexed video/audio writing with portable encoder contracts, bounded sink backpressure, gapless finalization, and deterministic seekable MP4 muxing.
- Add explicit CPU, native OpenGL, and browser WebGL frame transfer contracts with inspectable copy modes, conversion stages, context validation, strict fallback policies, and caller-safe resource ownership.
- Add the first portable implementation with checked timeline arithmetic, synchronized audio intervals, validated media buffers, capability and error types, asynchronous memory I/O, and native/WASM tests.
- Document complete planned native GL and browser WebGL workflows for synchronized video/audio reading and writing, including WebAssembly packaging and use.
- Initialize the project with its native and WebAssembly architecture, public API design, and Rust build scaffolding.
