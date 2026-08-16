# zvidlib architecture

## 1. Purpose and constraints

zvidlib is a Rust media library that provides frame-accurate, indexed video access and synchronized audio access for native and WebAssembly applications. The first complete vertical slice will read and write MP4-family containers carrying HEVC/H.265 or AV1 video and AAC audio, and transfer images through CPU memory, OpenGL, or WebGL.

This document is primarily a design contract. The repository implements the portable foundation—errors, limits, capability values, rational timeline arithmetic, synchronized audio intervals, validated CPU media buffers, byte I/O and encoder contracts, strict synchronized indexed output, and deterministic seekable MP4 muxing—while parsing, concrete codec backends, transfer, playback, and browser session layers remain planned.

The design is governed by these constraints:

1. `get(n)` means presentation frame `n`, even when decoding must begin at an earlier random-access point and reorder B/P frames.
2. `put(n, value)` produces a deterministic timeline and rejects accidental gaps, overlaps, and incompatible format changes unless the caller opts into a defined policy.
3. Sequential access reuses parser, decoder, reorder, and frame-cache state.
4. Native and browser builds share semantic APIs; platform adapters handle storage, scheduling, graphics, and audio differences.
5. Media logic does not assume a specific container, codec, graphics API, or I/O mechanism.
6. Third-party code is minimized, isolated behind traits, and evaluated for WASM support, binary size, safety, maintenance, and licensing.
7. Untrusted input is parsed with checked arithmetic and explicit resource limits. Unsafe code, if unavoidable for platform interop, stays in small audited backend modules.

## 2. Layered model

Dependencies point downward. Platform and codec integrations implement interfaces owned by the core rather than leaking their types upward.

```text
Rust API / generated JavaScript API
                 |
        session and stream API
     get(n), put(n), seek, finish
                 |
       timeline and media values
   frame index, rational time, planes
       /          |           \
 container     codec       transfer
 demux/mux   decode/encode  CPU/GL/WebGL
       \          |           /
       byte storage and platform runtime
```

### 2.1 Public session layer

An input or output session owns container state and exposes typed video and audio streams. The small common surface includes:

- `open(source, options)` and `create(sink, options)`;
- stream discovery and selection;
- video `get(frame_index)` and `put(frame_index, frame_source)`;
- audio `get(frame_index)`/`put(frame_index, samples)` plus sample-range operations;
- capability queries, metadata, cancellation, flushing, and finalization.

Operations are logically asynchronous on every target. Native Rust may later offer blocking convenience wrappers, but those wrappers must delegate to the same state machine so behavior remains consistent with JavaScript Promises.

A session is stateful and not implicitly concurrent. Independent sessions may run concurrently. Explicit prefetch and bounded pipelines provide parallelism without making ordered codec state racy.

### 2.2 Timeline layer

All time conversions use checked integer rational arithmetic. Floating-point seconds are display conveniences, never indexing authority.

Core concepts are:

- `FrameIndex`: zero-based presentation-order video frame;
- `Rational`: numerator/denominator with normalized sign and overflow checks;
- decode timestamp (DTS) and presentation timestamp (PTS) in the track time base;
- sample range: a half-open `[start, end)` range in an audio track's sample clock;
- edit mapping: movie timeline to track timeline, including empty edits and offsets.

Variable-frame-rate media is indexed from the ordered sample table. For video frame `n`, synchronized audio is the sample interval intersecting that frame's presentation interval. Rounding uses a documented boundary rule so adjacent requests neither duplicate nor lose samples. Encoder delay, AAC priming, end padding, and MP4 edit lists are retained and applied.

## 3. Frame-accurate reading

The MP4 demuxer builds or incrementally pages a compact sample index containing file offset, byte length, DTS, composition offset/PTS, duration, dependency flags, and random-access information. It does not assume decode and presentation order are equal.

For `get(n)`, the reader:

1. Returns the frame immediately if the presentation-indexed cache contains `n` in the requested representation.
2. Locates sample `n` and the nearest preceding valid random-access point using sync/dependency metadata and codec configuration.
3. Reuses the current decoder if its state can reach `n`; otherwise flushes it and seeks the byte source to that random-access point.
4. Feeds compressed samples in decode order, retaining decoded images in a reorder queue keyed by presentation identity.
5. Applies composition timestamps, edit mapping, and discard rules until exact presentation frame `n` is available.
6. Converts or transfers the frame to the requested CPU/GL/WebGL destination and records useful state for likely subsequent access.

HEVC CRA/IDR behavior, recovery points, leading pictures, and AV1 show-existing-frame semantics require codec-specific random-access validation. The container's sync flag alone is not always sufficient; a codec backend supplies dependency and reset information to the seek planner.

### 3.1 Cache policy

Each reader maintains separate bounded caches for:

- compressed byte ranges and parsed sample-index pages;
- decoder reference/reorder state;
- decoded frames in their native backend representation;
- optional converted CPU or GPU representations.

Budgets are expressed in bytes and frame counts, not an unbounded time window. The default policy favors the current frame, nearby presentation frames, the active group of pictures, and forward sequential reads. A large backward or unrelated seek evicts stale converted frames first. GPU resources are released on their owning context/runtime.

Prefetch is advisory and cancellable. It must not change which frame an indexed request returns, exceed configured resource limits, or conceal a decoder error needed by the caller.

## 4. Writing and synchronization

An output session accepts presentation-order frames and audio with explicit timing. The default indexed writer expects the next frame number; out-of-order or sparse writes require an option and a bounded staging policy.

The video encoder chooses decode order and reference structure. The muxer receives encoded samples with both DTS and PTS, writes composition offsets and sync/dependency tables, and finalizes durations only from exact timeline values. It may write a seekable file with metadata finalized in place or a fragmented MP4 stream when the sink cannot seek.

Audio input is accumulated into codec-sized blocks without changing its sample clock. Resampling is a separate opt-in transform. At finalization the writer records encoder priming and padding so decoded audio aligns with frame zero and ends at the intended boundary.

Backpressure propagates from sink to muxer, encoder, and caller. `finish` drains encoders, writes delayed B frames and audio packets, finalizes MP4 metadata, flushes the sink, and reports errors; dropping a writer is not a successful finalization mechanism.

## 5. Container subsystem

Container code is independent of codecs. Proposed responsibilities are split into:

- `ByteSource`/`ByteSink`: async random/sequential reads, writes, optional seek, length, and cancellation;
- `Probe`: bounded format detection without consuming caller-visible state;
- `Demuxer`: tracks, codec configuration, metadata, timed encoded samples, and seek indexes;
- `Muxer`: track declaration, timed encoded samples, metadata, fragmentation, and finalization.

The initial ISO Base Media File Format implementation covers the boxes necessary for ordinary and fragmented MP4, including movie/track metadata, sample description and timing tables, chunk/offset tables, sync/dependency information, edit lists, codec configuration, media data, and movie fragments. Unknown boxes are skipped safely and retained only when a preservation mode requests it.

Parsing is incremental and budgeted. Every size, offset, count, allocation, nesting level, and time conversion is validated. A declared box or sample may not address bytes outside its parent or source. Fuzzing and malformed fixtures are required before treating the parser as production-ready.

## 6. Codec subsystem

Codec interfaces operate on owned or lifetime-safe encoded packets and media values. Separate decoder and encoder factories advertise:

- codec identifiers and profiles;
- accepted/produced pixel or sample formats;
- resolution, channel, bit-depth, and rate limits;
- hardware/software and native/WASM availability;
- whether frames can be imported/exported through a particular graphics handle;
- configuration, drain, reset, and random-access behavior.

Container codec configuration is normalized before reaching a backend and serialized by the muxer without depending on backend-private types. This permits multiple implementations: browser WebCodecs, operating-system APIs, pure Rust/WASM codecs, or optional external adapters.

The initial codec priorities are HEVC/H.265 and AV1 video plus AAC audio. They are goals, not a promise that every browser exposes all three encoders/decoders. Capability discovery must distinguish unsupported codec, unsupported profile, invalid configuration, and unavailable hardware.

No codec implementation is automatically trusted with arbitrary allocation sizes. Backends receive limits and must validate decoded dimensions and formats before publishing a frame.

## 7. CPU, GL, and WebGL transfer

A video frame describes coded and display dimensions, plane layout, pixel format, alpha mode, pixel aspect ratio, orientation, and color metadata (primaries, transfer, matrix, range, and HDR metadata where present).

`FrameSource` and `FrameDestination` abstractions allow:

- owned or borrowed CPU planes with explicit stride;
- caller-supplied native GL textures/framebuffers;
- caller-supplied WebGL textures/framebuffers;
- backend-native images that can be converted or exported.

Graphics handles are tied to a context identity and execution owner. zvidlib never assumes a context is current, moves a non-transferable browser object between workers, or deletes caller-owned resources. Transfer commands execute through a context adapter supplied by the caller. Format conversion, scaling, orientation, and color conversion are explicit pipeline stages with inspectable cost.

Zero-copy is an optimization, not part of correctness. The capability model reports `Shared`, `GpuCopy`, `CpuCopy`, or `Unsupported`, and strict options can reject all but the requested classes.

## 8. Audio and playback adapters

Core audio APIs exchange timestamped sample buffers; they do not own an audio device. Native output and Web Audio integration are adapters above the core stream API.

A playback controller selects a monotonic master clock (normally the audio device clock), maps it to media time, requests the corresponding exact video frame, and schedules audio ahead within a bounded window. It may drop a late presentation frame but never silently substitute a different frame for `get(n)`. Seeking cancels queued work, resets decoder and audio scheduling state, applies the new edit mapping, and prerolls before resuming.

Recording adapters timestamp canvas captures and Web Audio/native audio buffers against one monotonic clock. Policies for variable capture cadence, duplicated frames, silence insertion, and discontinuities are explicit options rather than hidden repair.

## 9. Native and WebAssembly boundary

Portable core modules avoid operating-system handles, blocking I/O, native threads, and JavaScript types. Platform modules implement storage, task spawning, clocks, codec factories, and graphics transfer.

The JavaScript package will expose classes mirroring Rust sessions and streams. Boundary rules include:

- 64-bit frame and timestamp values use `BigInt` or validated safe-number conversions;
- async Rust operations become cancellable Promises where feasible;
- CPU frames can expose typed-array views only while backing memory is pinned and valid;
- browser objects such as `Blob`, `ReadableStream`, `VideoFrame`, WebGL contexts, and audio buffers remain platform adapter values;
- Rust errors become stable JavaScript error codes plus human-readable context.

The base WASM build does not require threads. Optional worker/thread acceleration must account for cross-origin isolation, shared memory availability, and object transfer rules. Feature selection must prevent native-only dependencies from compiling into browser builds.

## 10. Errors, observability, and limits

Public errors are categorized as invalid input, unsupported capability, malformed media, resource limit, I/O, codec, graphics/context, cancellation, invalid state, and internal invariant violation. Errors retain source context in Rust without making backend-specific types part of the stable API.

Sessions provide opt-in structured events for probing, seeking, cache activity, decoding/encoding, transfer fallback, synchronization, and mux finalization. Logging is never required for correctness and does not expose media contents by default.

Configurable limits cover dimensions, sample rates/channels, track count, metadata size, box depth/count, allocation bytes, cached frames, queued packets, decode work per seek, and output staging. Conservative browser defaults may differ from native defaults while preserving semantics.

## 11. Planned source boundaries

The initial implementation should remain one crate until independent release or dependency needs justify a workspace split:

```text
src/
  api/          sessions, streams, options, errors
  timeline/     rational time, indexes, edit mapping, synchronization
  media/        video frames, audio buffers, formats, metadata
  io/           byte source/sink traits and portable adapters
  container/    registry plus ISO BMFF demuxer and muxer
  codec/        traits, registry, normalized configuration
  transfer/     CPU conversion and graphics-neutral contracts
  platform/
    native/     native storage, runtime, codec, GL and audio adapters
    web/        browser storage, WebCodecs, WebGL and Web Audio adapters
  wasm_api/     JavaScript-facing wrappers and type conversion
```

Circular dependencies are forbidden. In particular, timeline and media values cannot depend on a container, codec, or platform implementation; container code cannot depend on GL/WebGL; and codec backends cannot directly drive playback.

## 12. Verification strategy

- Unit tests cover rational arithmetic, sample-table expansion, edit mapping, audio boundaries, cache eviction, and state transitions.
- Golden MP4 fixtures cover constant/variable frame rates, B-frame reordering, fragmented files, multiple tracks, edit lists, AAC priming, large offsets, and malformed structures.
- Codec conformance tests compare presentation hashes and timestamps from known vectors without depending on a single backend.
- Property tests generate valid and invalid container tables and assert bounds and round-trip invariants.
- Fuzz targets exercise box parsing, codec configuration parsing, and public byte-entry points.
- Cross-target integration tests run common semantics on native and headless browsers, including exact random/sequential seeks and synchronized recording.
- GPU tests verify ownership, context loss, fallback reporting, color conversion, and readback/upload across supported GL/WebGL versions.

Performance benchmarks measure sequential decode, cold random seek, warm nearby seek, CPU conversion, GPU transfer, memory high-water marks, and WASM bundle size. Performance work cannot weaken exact-frame or synchronization assertions.

## 13. Delivery sequence

1. Core types, errors, capability discovery, byte I/O, and timeline tests.
2. Read-only MP4 metadata/sample indexing with malformed-input and fuzz coverage.
3. Codec traits plus one decoder backend; exact-frame seeking and bounded caching.
4. CPU frames, then native GL and WebGL destinations.
5. AAC-aligned audio reads and playback adapters.
6. Encoders, MP4 muxing/finalization, and synchronized indexed writes.
7. Fragmented streaming, additional backends/codecs, and API stabilization.

Each stage must keep native and `wasm32-unknown-unknown` builds healthy. Public API stabilization follows end-to-end native and browser conformance rather than preceding it.
