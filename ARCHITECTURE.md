# zvidlib architecture

## 1. Purpose and constraints

zvidlib is a Rust media library that provides frame-accurate, indexed video access and synchronized audio access for native and WebAssembly applications. The first complete vertical slice will read and write MP4-family containers carrying HEVC/H.265 or AV1 video and AAC or Opus audio, and transfer images through CPU memory, OpenGL, or WebGL.

This document is primarily a design contract. The repository implements the portable foundation—errors, limits, capability values, rational timeline arithmetic, synchronized audio intervals, validated CPU media buffers, byte I/O, bounded ordinary/fragmented MP4 sample indexing, normalized codec factories, bounded exact-frame video decoding with portable conformance, accelerated Windows/Linux/macOS HEVC Main decode with a dependency-free software fallback, accelerated HEVC Main encode through Media Foundation on Windows and VideoToolbox on macOS, dependency-free native AV1 Main, VP8 and VP9 profile 0 decode and HEVC/AV1/VP8/VP9 encode backends (`native_hevc_video_decoder_factory`, `native_hevc_video_encoder_factory`, `native_av1_video_decoder_factory`, `native_vp9_video_decoder_factory`, `native_vp8_video_decoder_factory`, `native_av1_video_encoder_factory`, `native_vp8_video_encoder_factory`, `native_vp9_video_encoder_factory`), exact AAC, Opus and Vorbis packet/sample reads with pure-Rust Opus and Vorbis decoders and encoders, audio-clock playback control and native/Web Audio adapter contracts, encoder contracts, CPU/GL/WebGL transfer contracts, strict synchronized indexed output, deterministic seekable MP4 muxing, and bounded WebM sample indexing with seekable AV1, VP8 and VP9 WebM muxing—while a concrete AAC encoder and audio-device bindings remain planned.

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

Variable-frame-rate media is indexed from the ordered sample table. For video frame `n`, synchronized audio is the sample interval intersecting that frame's presentation interval. Rounding uses a documented boundary rule so adjacent requests neither duplicate nor lose samples. Encoder delay, AAC priming, Opus pre-skip, end padding, and MP4 edit lists are retained and applied.

## 3. Frame-accurate reading

The MP4 demuxer builds or incrementally pages a compact sample index containing file offset, byte length, DTS, composition offset/PTS, duration, dependency flags, and random-access information. It does not assume decode and presentation order are equal.

For `get(n)`, the reader:

1. Returns the frame immediately if the presentation-indexed cache contains `n` in the requested representation.
2. Locates sample `n` and the nearest preceding valid random-access point using sync/dependency metadata and codec configuration.
3. Reuses the current decoder if its state can reach `n`; otherwise flushes it and seeks the byte source to that random-access point.
4. Feeds compressed samples in decode order, retaining decoded images in a reorder queue keyed by presentation identity.
5. Applies composition timestamps, edit mapping, and discard rules until exact presentation frame `n` is available.
6. Converts or transfers the frame to the requested CPU/GL/WebGL destination and records useful state for likely subsequent access.

HEVC CRA/IDR behavior, recovery points, leading pictures, AV1 show-existing-frame semantics, VP8 hidden frames, and VP9 hidden frames and superframes require codec-specific random-access validation. A VP8 sample holds exactly one frame; a hidden frame (`show_frame` = 0, usually an alternate reference coded ahead of the frames that use it) is decoded and updates the references but produces no picture, so a container must give it an identity that is never requested as a presentation frame, and a seek to any later frame decodes through it from the preceding key frame. The container's sync flag alone is not always sufficient; a codec backend supplies dependency and reset information to the seek planner.

### 3.1 Cache policy

Each reader maintains separate bounded caches for:

- compressed byte ranges and parsed sample-index pages;
- decoder reference/reorder state;
- decoded frames in their native backend representation;
- optional converted CPU or GPU representations.

Budgets are expressed in bytes and frame counts, not an unbounded time window. The default policy favors the current frame, nearby presentation frames, the active group of pictures, and forward sequential reads. A large backward or unrelated seek evicts stale converted frames first. GPU resources are released on their owning context/runtime.

Prefetch is advisory and cancelable. It must not change which frame an indexed request returns, exceed configured resource limits, or conceal a decoder error needed by the caller.

### 3.2 Seek latency

A seek to any position of any track must be constant time and must complete in under 50 ms in the worst case. Constant time here means independent of the seek distance and of the track length: seeking to the last frame of a two-hour track costs what seeking to the second frame costs, and neither is allowed to be proportional to the number of frames between the current position and the target.

This is a requirement on `seek`, which answers *what is at this position of the timeline*, and not on `get(n)`, which answers *exactly which frame this is*. The two cannot have the same bound. A frame in the middle of a long group of pictures depends on every frame back to its random-access point, and decoding them is the track's cost, not the decoder's: on a 1080p HEVC track coded as a single group of pictures, the last frame is roughly 700 reference decodes away from the only place a decode can start, which is over a second on hardware that manages 700 pictures a second. No arrangement of one decoder reaches 50 ms from there, so `seek` may not be defined as a synchronous exact decode.

A seek is therefore answered from pictures that are already decoded, and never by decoding:

- the presentation cache, when the requested frame is still in it;
- otherwise a bounded seek preview index — one downscaled picture every N frames, populated by a background pass over the track on a decoder of its own, sized by a memory budget rather than by a fixed count, and answering the nearest position it has reached while it is still being built;
- otherwise nothing, reported as such, so the caller falls through to its own exact request rather than blocking on the fast tier.

The preview index is part of the library rather than of each application: every caller that scrubs a timeline needs it, and a caller that has to build its own has a seek that does not meet this requirement until it does. The reader reaches it through a trait rather than by naming it, because the reader is portable and an index whose pass owns a thread is not; a browser-side source implements the same lookup against whatever it can decode ahead.

Exactness is unaffected. A preview is explicitly not the frame that was asked for, is labeled with the frame it is of, and never substitutes for `get(n)`, which continues to return exactly frame `n` or an error (section 8).

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

WebM, the Matroska subset browsers record and play, is the second container. `probe_container` tells the two apart by signature alone - an EBML header with a `webm` or `matroska` `DocType`, or an `ftyp`/`moov` box - so a file name or MIME type never decides how bytes are read. `WebmDemuxer` scans the Segment once and produces the same `Track` decode-order sample index the MP4 demuxer does, so every consumer of an index - `ExactFrameReader`, the preview tier, the browser decode session - reads either container without knowing which it is. It reads only metadata elements (`Info`, `Tracks`, `Cues`) whole, and only the header and lacing of a block, never its payload. It accepts unknown-size Segments and Clusters, which is how a live `MediaRecorder` capture is written, ending an unknown-size Cluster at the next Segment child; SimpleBlocks and BlockGroups; Xiph, EBML and fixed-size lacing; and any `TimestampScale`, counting in its ticks when it divides a second and in nanoseconds otherwise. Block timestamps are presentation times, and a frame's duration comes from its `BlockDuration`, the next block, the track's `DefaultDuration`, or the segment `Duration`, in that order. `Cues`, when present, mark random-access points and answer `seek_point`; without them the scanned keyframe flags do. Every element header counts against an element budget, and every element read whole against a size limit.

`WebmMuxer` mirrors the seekable MP4 muxer: payload is written as it arrives, a Cluster opens at each video keyframe (or when a 16-bit relative block timestamp would overflow), and `finish` seeks back to fill in the Segment and Cluster sizes, the `Duration` and a fixed-width `SeekHead`, then appends `Cues` with one cue per keyframe. The `CodecPrivate` mapping is per codec - `V_AV1` carries the `av1C` record without its box header, and the demuxer wraps it back so decoders configure identically from either container - `V_VP8` carries none, because a VP8 key frame holds everything a decoder needs, and `V_VP9` carries the `vpcC`'s profile, level, bit depth and chroma subsampling as the WebM VP9 mapping's `CodecPrivate` features, which the demuxer turns back into a `vpcC` box. WebM permits only Vorbis or Opus audio, so WebM output refuses an AAC track, and WebM input skips the audio tracks it has no reader for. An Opus or Vorbis track's trims are WebM's own: its `CodecDelay` is the priming and its last block's `DiscardPadding` the end padding. The padding is only known once the encoder drains, after its last packets, so the muxer holds each audio track's newest block back until a later block, the track's gapless trim, or `finish` writes it, and in a file without video it opens a Cluster every five seconds and cues each one's first audio block.

## 6. Codec subsystem

Codec interfaces operate on owned or lifetime-safe encoded packets and media values. Separate decoder and encoder factories advertise:

- codec identifiers and profiles;
- accepted/produced pixel or sample formats;
- resolution, channel, bit-depth, and rate limits;
- hardware/software and native/WASM availability;
- whether frames can be imported/exported through a particular graphics handle;
- configuration, drain, reset, and random-access behavior.

Container codec configuration is normalized before reaching a backend and serialized by the muxer without depending on backend-private types. This permits multiple implementations: browser WebCodecs, operating-system APIs, pure Rust/WASM codecs, or optional external adapters.

The initial codec priorities are HEVC/H.265 and AV1 video plus AAC audio; VP8 decode and encode follow for WebM: decode through WebCodecs' `vp8` decoder in the browser with the pure-Rust decoder as the native backend and browser fallback, and encode through WebCodecs' `vp8` encoder in the browser and a pure-Rust encoder natively. The native VP8 encoder reconstructs every macroblock with the decoder's own reconstruction and loop filter, so its references are the decoder's pictures by construction rather than by a parallel implementation that could drift; it codes key frames and inter frames from the last frame only, never golden or alternate references, and each frame's adapted token probabilities are not kept for the next, so a decoder starts every frame from the defaults. Opus and Vorbis audio sit alongside AAC. They are goals, not a promise that every browser exposes every encoder/decoder. Capability discovery must distinguish unsupported codec, unsupported profile, invalid configuration, and unavailable hardware.

Audio codecs map onto containers as follows. AAC is carried in MP4 `mp4a`/`esds`. Opus is carried in MP4 as an `Opus` sample entry with a `dOps` box, always on a 48 kHz sample clock because Opus always decodes at 48 kHz; its pre-skip is informative there and the edit list does the trimming, and each packet's decoded length is read from its own table of contents, since muxers shorten the last sample's table duration to trim the end. WebM carries Opus's RFC 7845 `OpusHead` and Vorbis's three Xiph-laced setup headers as `CodecPrivate`, a pre-skip as `CodecDelay` and an end trim as the last block's `DiscardPadding`. Vorbis has no widely supported MP4 mapping, so the MP4 muxer refuses it. A Vorbis packet's decoded length depends on its own block size and the previous packet's, which the setup header's modes determine, so the configuration parser reads them; a stream's first packet decodes no samples.

Every audio codec reads through one exact sample reader, which maps presentation ranges through priming, padding and edits onto packets and decodes a codec-specific preroll ahead of a seek: one packet for Vorbis, whose packets depend only on the overlap with the previous one, and 320 ms for Opus, by when its prediction state matches a continuous decode. Preroll output is never cached for a later request. The browser build drives the same reader with `WebCodecs` `AudioDecoder` where the browser supports the track, from a freshly configured decoder for each read, and checks the browser's output fills the packets' intervals exactly, falling back to the pure-Rust decoder where it does not. Audio encoders are either the platform's (AAC) or faithful ports of the reference encoders (`opus-pure` for libopus, and zvidlib's own port of libvorbis, bit-identical to it), never a codec design of zvidlib's own. AAC decoding is the platform's too wherever the platform has a decoder, as in the browser: AudioToolbox on macOS and Media Foundation's AAC decoder MFT on Windows. Linux and the other native targets have none, and decode through Symphonia's pure-Rust AAC decoder, the one third-party AAC decoder zvidlib still carries. Media Foundation emits each access unit's PCM only once the next one arrives, so the native decoder starts it clean for every unit, feeds the unit with the one before it, and pushes the frame out by repeating the unit; an AAC-LC frame depends only on those two, so the exact sample reader sees the same packet-aligned intervals as from any other decoder.

VP9 profile 0 decoding follows the same pattern as AV1: an MP4 `vp09` sample entry's `vpcC` box is the normalized configuration, and a WebM `V_VP9` track's optional `CodecPrivate` is normalized into the same box, the browser decodes through `WebCodecs` with a `vp09.PP.LL.DD` codec string where it can, and `native_vp9_video_decoder_factory` is the pure-Rust decoder everywhere else. One VP9 sample (an MP4 sample or a WebM block) is one chunk as the VP9 ISO-BMFF binding defines it: hidden frames (alternate references) travel in a superframe with the frame it shows, and a `show_existing_frame` header shows a reference again, so every sample still shows exactly one frame and exact-frame access maps samples to presentation frames one to one. Because VP9's decoding process is defined by its reference decoder, the pure-Rust decoder is verified against libvpx's own per-frame digests of its profile 0 test vectors rather than against a second independent decoder. Its inverse transforms, inter and intra prediction and loop filter dispatch at run time to SSE4.1, AVX2 or NEON kernels (`crates/zvidlib-vp9-decoder/src/vp9_simd/`, the `vp9_decode` site of `zvidlib::simd`) that are bit-exact with the scalar code they replace, which remains the fallback on every other target.

VP9 profile 0 (`Codec::Vp9`, `CodecProfile::Vp9Profile0`) is an output codec as well. MP4 carries it as a `vp09` sample entry whose `vpcC` box records the profile, level, bit depth, chroma subsampling, range and color description; VP9 has no out-of-band parameter sets, so the box is all a decoder needs beside the frames. `native_vp9_video_encoder_factory` (`crates/zvidlib-vp9-encoder/src/`) is a pure-Rust encoder that follows the AV1 encoder's shape: `frame.rs` makes every coding decision and writes the boolean-coded headers and tiles, `dsp.rs` holds the forward transforms, intra predictors and 8-tap motion compensation, which must match the VP9 decoding process bit for bit because later blocks and frames predict from the encoder's own reconstruction, and reconstructs with the native decoder's own inverse transforms, and `tables.rs` holds default probabilities from the specification; the coefficient probabilities and scans of every transform size are the decoder's tables. It splits each superblock into 64x64 down to 8x8 blocks with transforms from 4x4 to 32x32, all chosen by rate and distortion, with key frames and inter frames against the previous frame whose motion vectors are refined to quarter or eighth samples. Frames adapt their probabilities backwards (`refresh_frame_context` without frame-parallel decoding mode) and take the previous frame's motion vectors as candidates, so `context.rs` keeps the encoder's frame context and symbol counts in step with the decoder's, frame to frame and across the reset at each key frame, and rates are costed with the probabilities each frame actually codes with; a configuration flag makes every frame error resilient with the default probabilities instead. It runs the decoder's own loop filter (`vp9_dec::loopfilter`) over each reconstruction at the frame level libvpx's `search_filter_level` would pick, so later frames predict from the deblocked picture the decoder sees. `Prefer` and `Require` select a hardware VP9 encoder through the HEVC encoder's platform backends instead, generalized over the output codec: a hardware Media Foundation VP9 MFT on Windows (`crates/zvidlib-hardware/src/windows_mf_encoder.rs`) and VideoToolbox on macOS (`crates/zvidlib-hardware/src/videotoolbox_encoder.rs`). Both are rate controlled by quality from the same `base_q_idx`, encode one throwaway priming frame to learn the color space their key frames signal, and build the `vpcC` from it, so the track declares what the bitstream carries; a later key frame that signals anything else is an error, and a sample is sync exactly when its first frame is a key frame. The browser encodes VP9 through `WebCodecs` (`vp09.00.LL.08`) and builds the `vpcC` from the first key frame's color fields, as it builds `av1C` from the AV1 sequence header. The browser decodes VP9 through `WebCodecs`, which takes no description for it.

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

Core audio APIs exchange timestamped sample buffers; they do not own an audio device. Native output and Web Audio integration are adapters above the core stream API, and the playback controller reads AAC, Opus and Vorbis tracks through the same exact sample reader.

A playback controller selects a monotonic master clock (normally the audio device clock), maps it to media time, requests the corresponding exact video frame, and schedules audio ahead within a bounded window. It may drop a late presentation frame but never silently substitute a different frame for `get(n)`. Seeking cancels queued work, resets decoder and audio scheduling state, applies the new edit mapping, and prerolls before resuming.

Recording adapters timestamp canvas captures and Web Audio/native audio buffers against one monotonic clock. Policies for variable capture cadence, duplicated frames, silence insertion, and discontinuities are explicit options rather than hidden repair.

## 9. Native and WebAssembly boundary

Portable core modules avoid operating-system handles, blocking I/O, native threads, and JavaScript types. Platform modules implement storage, task spawning, clocks, codec factories, and graphics transfer.

The `web` feature exposes classes mirroring Rust sessions, streams, options, and media values through `wasm-bindgen`. Boundary rules include:

- 64-bit frame and timestamp values use `BigInt` or validated safe-number conversions;
- async Rust operations become cancelable Promises where feasible;
- CPU frames and audio buffers cross the boundary as owned copies; returned typed arrays are snapshots rather than views into growable WASM memory;
- browser objects such as `Blob`, `ReadableStream`, `VideoFrame`, WebGL contexts, and audio buffers remain platform adapter values;
- Rust errors become stable JavaScript error codes plus human-readable context.

Blob and buffer inputs are copied into WASM-owned storage. `ReadableStream<Uint8Array>` inputs are consumed with a bounded allocation, release their reader locks, and can be canceled with `AbortSignal`. Finalized browser output is copied into a browser-owned `Blob`. Session and stream handles are present even when a backend is unavailable; such operations reject with `UNSUPPORTED` instead of returning placeholder media.

The base WASM build does not require threads. Optional worker/thread acceleration must account for cross-origin isolation, shared memory availability, and object transfer rules. Feature selection must prevent native-only dependencies from compiling into browser builds.

## 10. Errors, observability, and limits

Public errors are categorized as invalid input, unsupported capability, malformed media, resource limit, I/O, codec, graphics/context, cancellation, invalid state, and internal invariant violation. Errors retain source context in Rust without making backend-specific types part of the stable API.

Sessions provide opt-in structured events for probing, seeking, cache activity, decoding/encoding, transfer fallback, synchronization, and mux finalization. Logging is never required for correctness and does not expose media contents by default.

Configurable limits cover dimensions, sample rates/channels, track count, metadata size, box depth/count, allocation bytes, cached frames, queued packets, decode work per seek, and output staging. Conservative browser defaults may differ from native defaults while preserving semantics.

## 11. Source boundaries

The library is a Cargo workspace (#604), so a change to one codec recompiles, tests and benchmarks that codec and the crates that depend on it rather than the whole library. `zvidlib`, the root package, is still the only crate a user depends on: it owns the public API, the session, output and playback layers and the wasm/web bindings, and re-exports every subcrate's public items under the paths they had when they were modules of one crate. The subcrates are published alongside it, at the same version.

```text
src/                       zvidlib: lib.rs re-exports, simd (dispatch-site report), output,
                           playback, previews, native audio, wasm/web bindings
crates/
  zvidlib-core/            errors and limits, timeline, media values, I/O, codec and audio
                           traits, transfer contracts, the VP9 codec configuration record,
                           and the process-wide SIMD override and vector types
  zvidlib-color/           YUV/RGBA conversion shared by the video decoders, the HEVC and
                           VP8 encoders and the hardware backends
  zvidlib-container/       MP4 and WebM muxers and demuxers, probing, codec strings,
                           cover art, and the conformance harness
  zvidlib-opus-syntax/     Opus configuration records and packet timing
  zvidlib-opus/            Opus decoding and encoding
  zvidlib-vorbis-syntax/   Vorbis header parsing and packet timing
  zvidlib-vorbis-decoder/  Vorbis decoding
  zvidlib-vorbis-encoder/  the libvorbis encoder port
  zvidlib-aac-encoder/     platform AAC encoding
  zvidlib-av1-syntax/      AV1 OBU and configuration record parsing
  zvidlib-av1/             AV1 coding tools and SIMD kernels shared by the encoder and the
                           reference decoders
  zvidlib-av1-decoder/     the AV1 Main-profile software decoder
  zvidlib-av1-encoder/     the AV1 encoder
  zvidlib-vp8/             the VP8 decoder and encoder, which share reconstruction
  zvidlib-vp9-syntax/      VP9 frame inspection, picture output and level selection
  zvidlib-vp9-decoder/     the VP9 software decoder
  zvidlib-vp9-encoder/     the VP9 encoder
  zvidlib-hevc-syntax/     HEVC bit reading and writing, CABAC, parameter sets, slice headers
  zvidlib-hevc/            the HEVC engine and SIMD kernels shared by the decoder and encoder
  zvidlib-hevc-decoder/    the HEVC software decoder
  zvidlib-hevc-encoder/    the HEVC encoder
  zvidlib-hardware/        NVDEC, Media Foundation and VideoToolbox backends
  zvidlib-build/           the build-script helper for the Swift runtime search path
  zvidlib-bench-support/   shared benchmark helpers (development only, unpublished)
```

`zvidlib` depends on each codec crate through a Cargo feature per codec and direction, and on the containers, the core and the syntax crates unconditionally, so a build names only the codecs it uses (#635). Each codec crate's factory selects between its software implementation and the hardware backends, behind the codec crate's `hardware` feature, so `zvidlib-hardware` sits below the codec crates and depends only on the syntax and color crates; the one thing a backend needs from a software codec, VideoToolbox's replay of a hidden VP9 frame, is handed to it by the VP9 decoder as a function. Tests, fixtures and benchmarks live in the crate they exercise; the root package keeps the API-level and CI-level ones.

Circular dependencies are forbidden. In particular, timeline and media values cannot depend on a container, codec, or platform implementation; container code cannot depend on GL/WebGL; and codec backends cannot directly drive playback. The crate graph enforces the first two.

## 12. Verification strategy

- Compressed decoder backends must land with checked-in `VideoDecoderConformanceVector` inputs and expected canonical `FrameDigest` values. The shared runner verifies every presentation frame sequentially, in reverse, and with alternating distant seeks so reorder, reset, and exact-frame behavior cannot be special-cased to linear playback.
- Compressed encoder backends must land with checked-in `VideoEncoderConformanceVector` source frames and an explicit PSNR floor. The shared runner validates configuration and packet timing, then round-trips the output through a decoder that has independently passed decoder conformance; encoder and decoder implementations must not share a private nonstandard bitstream path. The VP9 encoder's vectors round-trip through both `native_vp9_video_decoder_factory()` and libvpx (or FFmpeg's VP9 decoder) behind the `VideoDecoder` contract, and its unit tests require libvpx and FFmpeg to reproduce the encoder's reconstruction exactly, frame for frame.
- Unit tests cover rational arithmetic, sample-table expansion, edit mapping, audio boundaries, cache eviction, and state transitions.
- Golden MP4 fixtures cover constant/variable frame rates, B-frame reordering, fragmented files, multiple tracks, edit lists, AAC priming, large offsets, and malformed structures.
- Codec conformance tests compare presentation hashes and timestamps from known vectors without depending on a single backend.
- Property tests generate valid and invalid container tables and assert bounds and round-trip invariants.
- Fuzz targets exercise box parsing, codec configuration parsing, and public byte-entry points.
- Cross-target integration tests run common semantics on native and headless browsers, including exact random/sequential seeks and synchronized recording.
- GPU tests verify ownership, context loss, fallback reporting, color conversion, and readback/upload across supported GL/WebGL versions.

Performance benchmarks measure sequential decode, cold random seek, warm nearby seek, CPU conversion, GPU transfer, memory high-water marks, and WASM bundle size. Performance work cannot weaken exact-frame or synchronization assertions.

The criterion benchmark suite lives in `benches/` and is documented in `benches/README.md`: a single `harness = false` bench target over shared, once-per-process fixtures, with the bundled 1080p sample gated behind a long-running group. Every criterion group name carries the arm it was measured under (`hevc_decode/simd=off` or `hevc_decode/simd=on`) from the additive, off-by-default `simd` cargo feature, so the two builds record separately and stay comparable.

## 13. Delivery sequence

1. Core types, errors, capability discovery, byte I/O, and timeline tests.
2. Read-only MP4 metadata/sample indexing with malformed-input and fuzz coverage.
3. Codec traits plus one decoder backend; exact-frame seeking and bounded caching.
4. CPU frames, then native GL and WebGL destinations.
5. AAC-aligned audio reads and playback adapters.
6. Encoders, MP4 muxing/finalization, and synchronized indexed writes.
7. Fragmented streaming, additional backends/codecs, and API stabilization.

Each stage must keep native and `wasm32-unknown-unknown` builds healthy. Public API stabilization follows end-to-end native and browser conformance rather than preceding it.

## Appendix: backend and API reference

The detail behind the support tables in the [README](README.md): what each browser entry point accepts, how WebM is read and written, and what each native encoder and decoder backend does. The rustdoc API documentation is the reference for every item named here.

### A.1 Browser API boundary

The `web` feature exposes `MediaInput`, `MediaOutput`, `Playback`, `VideoStream`, `AudioStream`, `OpenOptions`, `CreateOptions`, `PlaybackOptions`, `PreviewIndex`, `PreviewOptions`, `Preview`, `FrameIndex`, `Timestamp`, `Rational`, `SampleRange`, `VideoFrame`, and `AudioBuffer` through `wasm-bindgen`.

`MediaInput.open` accepts a `Blob`, `ReadableStream<Uint8Array>`, `ArrayBuffer`, or typed-array view. It consumes streams, always releases its reader lock, and supports cancellation through `OpenOptions.signal`. Input bytes are copied into owned WebAssembly storage; `bytes()` returns a fresh JavaScript snapshot rather than a view into growable WebAssembly memory.

An input `AudioStream` reads exact sample ranges: `getRange(start, end)` resolves to an `AudioBuffer` holding exactly the half-open range `[start, end)` of the track's presentation samples, after its priming, end padding and edit list, and `sampleCount()` is one past the last of them. AAC and Opus tracks decode through the browser's `WebCodecs` `AudioDecoder` where it supports them, and Opus falls back to zvidlib's own decoder where it does not or where the browser's output does not line up with the track. `decoderConfig()` returns a track's `WebCodecs` `AudioDecoderConfig` for a caller driving its own decoder with `packet()`.

`MediaOutput.finish()` returns a browser-owned `Blob` with the configured MIME type. `writeEncodedChunk()` remains the raw byte-sink boundary for a caller supplying its own already-muxed container bytes. Indexed `put` on any video track encodes through a built-in `WebCodecs` bridge (AV1 Main, HEVC Main, VP8 or VP9 profile 0; `CreateOptions.videoCodec` selects `"av1"` (the default), `"hevc"`, `"vp8"` or `"vp9"`) and mixes every track's samples into a real multi-track MP4 at `finish()`; call `videoEncodeSupport(hardware, codec)` first to check whether the current browser can encode the selected codec. A VP9 track is written as a `vp09` sample entry whose `vpcC` takes its color description from the encoder's first key frame. Input frames accept `VideoFrame.rgba()`, `VideoFrame.bgra8()`, or `VideoFrame.yuv420p8()`. Track 0 of an audio stream likewise encodes through a `WebCodecs` `AudioEncoder` bridge from interleaved `f32` PCM: AAC-LC by default, or Opus with `CreateOptions.audioCodec = "opus"`, which takes 48 kHz input, the rate Opus decodes at, and writes the encoder's pre-skip and end padding into the MP4's edit list so the track reads back with exactly the samples that were put. Call `audioEncodeSupport(hardware, codec)` first; it reports `false` for `"vorbis"`, which no browser encodes and MP4 cannot carry. `finish()` muxes whichever tracks were used into one MP4. Other audio track indices and other codecs/profiles remain unsupported until their backends land. `setCoverArt(bytes, "image/jpeg" | "image/png")`, called any time before `finish()`, embeds an already-encoded picture as iTunes-style `covr` metadata, which Windows Explorer, Finder and most players show as the file's thumbnail without decoding the video. Without one, `finish()` generates the cover itself from video frame 4 of the first video track, shrunk to at most 512 pixels on its longest edge and encoded as PNG, or from the last frame of a shorter stream; set `CreateOptions.coverFrame` to another zero-based index, or to `null` for no generated cover.

All exported 64-bit frame, sample, and timestamp values return JavaScript `BigInt`. Inputs accept `BigInt` across the full Rust range or validated `Number` values only within JavaScript's safe-integer range. Rust and boundary failures reject with native `Error` instances named `ZvidError`; their stable `code` values can be read directly or with `errorCode(error)`.

Input `VideoStream` handles also expose `frameDuration(index)`, which resolves to that presentation
frame's MP4 duration in milliseconds. This timing query is independent of codec availability, so
browser applications can pace fallback rendering even when `get(index)` reports `UNSUPPORTED`.

`video.previews(options)` builds the seek preview tier `ARCHITECTURE.md` section 3.2 requires a
seek to be answered from: one downscaled picture every stride frames, on a decode session of its
own. The pass has no thread to run on in a browser, so the caller advances it a preview at a time
with `await index.step(signal)` from `requestIdleCallback` or a `requestAnimationFrame` slice, and
`index.nearest(frame)` answers from whatever it has reached so far. That lookup is *synchronous*
and never decodes - it returns a `Preview` carrying the picture and the frame it is actually of, or
`null` - which is what lets it stay inside `seekLatencyBudgetMs()` however far a drag jumped.

### A.2 WebM

WebM is read and written alongside MP4 through the same indexed `get(n)`/`put(n)` API. `MediaInput.open` tells the two apart by the bytes themselves, never by a file name or MIME type: an EBML header whose `DocType` is `webm` or `matroska` is WebM, and `input.container` reports `"webm"`, `"mp4"`, or `null`. A WebM input's AV1 video tracks then answer `get(n)`, `frameDuration(n)`, `randomAccessPoints()` and `previews()` exactly as an MP4's do. `new CreateOptions("webm")` makes `MediaOutput.finish()` mux the encoded video into a seekable WebM and return a `video/webm` Blob; `supportedContainers()` lists both names.

WebM permits only VP8, VP9 or AV1 video and Vorbis or Opus audio, so WebM output is AV1, VP8 or VP9 video with Opus or Vorbis audio: `CreateOptions.audioCodec` defaults to `"opus"` on a WebM output and refuses `"aac"`, `videoCodec = "vp8"` encodes VP8 through the browser's `WebCodecs` `VideoEncoder` (`videoEncodeSupport(hardware, "vp8")` reports whether it can) and is refused with `UNSUPPORTED` on an `"mp4"` output, since MP4 has no widely supported VP8 mapping, and `setCoverArt()` and `videoCodec = "hevc"` reject with `UNSUPPORTED`. On input, Opus and Vorbis tracks read through `AudioStream.getRange()` with their `CodecDelay` and `DiscardPadding` trims applied, and other audio tracks are skipped rather than refused, so a browser `MediaRecorder` capture still opens for its video; VP8 video decodes through the browser's `WebCodecs` `vp8` decoder, or zvidlib's own VP8 decoder where `WebCodecs` cannot, and VP9 profile 0 video through `WebCodecs` with a `vp09` codec string, or zvidlib's own VP9 decoder where `WebCodecs` cannot. A VP8 frame the encoder hid (an alternate reference stored as a block of its own) is currently indexed as a frame, and asking for that frame fails. A VP9 track is written as `V_VP9`, with the profile, level, bit depth and chroma subsampling of its `vpcC` as the `CodecPrivate` features, and reads back with an equivalent `vpcC`.

Natively, `WebmDemuxer::open` builds the same `Track` sample index `Mp4Demuxer::open` does, so `to_encoded_video_samples` and `ExactFrameReader` read a WebM track unchanged. It handles unknown-size Segments and Clusters (as live recorders write them), SimpleBlocks and BlockGroups, Xiph, EBML and fixed-size lacing, and any `TimestampScale`; `WebmDemuxer::seek_point` starts a decode from the file's `Cues` when it has them and from the scanned keyframes when it does not. `WebmMuxer` takes the same `Mp4TrackConfig` declarations `Mp4Muxer` does, for AV1, VP8 and VP9 video (a `V_VP8` track carries no `CodecPrivate`, so its `decoder_config` is empty) and Opus and Vorbis audio, writes payload bytes as samples arrive, and at `finish` fills in the Segment and Cluster sizes, the `Duration`, a `SeekHead`, and `Cues` with one cue per keyframe, which is what lets Chrome and Firefox seek the file. Samples are written in one presentation-time order across tracks, because a WebM Cluster interleaves them. An Opus or Vorbis track is indexed into the same `Track` an MP4 audio track is - an Opus `OpusHead` rewritten as the `dOps` box `opus_config` reads, a Vorbis `CodecPrivate` as stored for `vorbis_config` - and `WebmDemuxer::audio_timing` turns its `CodecDelay` and last `DiscardPadding` into the priming and padding `AudioSampleReader` trims. `WebmMuxer` writes an Opus track's pre-skip as `CodecDelay` with an 80 ms `SeekPreRoll`, and the padding `set_audio_gapless` declares as the track's last block's `DiscardPadding`, holding each audio track's newest block back until it knows whether that block is the last; FFmpeg decodes zvidlib's Opus WebM to exactly the encoded length. `probe_container` detects either container from a `ByteSource`, and `container_capabilities` reports both.

### A.3 Writer core and native encoders

zvidlib writes **no audio codec of its own design**. Every `AudioEncoder` it ships is either the platform's or a faithful port of the codec's reference encoder, because a mediocre encoder under this crate's name is harder for callers to route around than no encoder at all, and with a port the quality to defend is the reference's. `native_aac_audio_encoder_factory()` delegates to AudioToolbox on macOS and to Media Foundation on Windows, and reports `HardwareUnavailable` elsewhere rather than falling back to an encoder of its own: a pure-Rust AAC-LC encoder means owning a filter bank, a psychoacoustic model, and rate control with no open reference to port. `native_opus_audio_encoder_factory()` runs `opus-pure`, a pure-Rust port of libopus, and `native_vorbis_audio_encoder_factory()` runs zvidlib's own port of the libvorbis 1.3.7 encoder, whose packets are bit-identical to libvorbis's; both run on every target. `MediaOutput` muxes an audio track given any `AudioEncoder`, so a caller with an encoder of its own still plugs it in.

The implemented portable writer core exposes `VideoEncoder` and `AudioEncoder` contracts plus `MediaOutput`. An encoder backend declares its MP4 codec configuration and exact output timescale, then returns `EncodedSample` values with DTS, PTS, duration, sync, and dependency metadata. `MediaOutput::put_video` accepts the shared CPU/GL/WebGL `FrameSource`; it and `put_audio` enforce zero-based consecutive indices and exact frame-aligned audio ranges. `finish` drains both encoders, records audio priming and padding in an edit list, finalizes sample indexes, and flushes the seekable `ByteSink`. `MediaOutput::set_cover_art` (or `Mp4Muxer::set_cover_art`) embeds a caller-encoded JPEG or PNG `CoverArt` in `moov/udta/meta/ilst/covr` any time before `finish`, so a recorder can pick a frame after capture; operating-system file browsers show it as the thumbnail without an HEVC or AV1 decoder, output without cover art is unchanged, and `Mp4Demuxer::cover_art` reads it back. When no cover is set, `MediaOutput` generates one: `OutputOptions::cover_source` defaults to `CoverSource::Frame(4)`, and the writer shrinks that presentation-order frame to at most 512 pixels on its longest edge as frames are written, then encodes it as a PNG at `finish` with no extra dependency. A stream with fewer frames uses its last frame, a frame supplied as a GPU resource gives no cover, and `CoverSource::None` turns the generated cover off. The capture holds one thumbnail, at most 768 KiB, and costs nothing after the chosen frame. `CoverArt::from_video_frame` makes the same thumbnail from any CPU frame. `Mp4Muxer` never generates a cover, so its output without cover art is unchanged.

`native_av1_video_encoder_factory()` supplies a dependency-free software backend for 8-bit monochrome AV1 Main profile. It accepts `Gray8` CPU frames, emits one independently decodable keyframe access unit per input frame, and generates the matching standardized `av1C` box. `VideoEncoderConfig::configuration` selects the quantization profile: empty encodes losslessly with the 4x4 WHT (the default, and the output verified against an independent ffmpeg decode), and a single byte carrying a nonzero `base_q_idx` encodes non-lossless through the forward DCT with a per-block transform-size, transform-type, and partition decision. Non-lossless streams are interchange-grade: they round-trip through `native_av1_video_decoder_factory()` and decode through an independent ffmpeg 7.1 (dav1d) within an asserted distortion bound across several quantizers, frame sizes, and content patterns. `VideoEncoderConfig::timescale` and `frame_duration` define the exact constant-rate clock used for emitted DTS, PTS, and duration values. Unsupported color formats, hardware requirements, malformed private configuration, out-of-level rates/dimensions, and resource-limit violations are rejected during capability discovery or creation.

`native_vp8_video_encoder_factory()` supplies a dependency-free software VP8 encoder. It accepts limited-range `Rgba8` or `Bgra8` CPU frames of any size up to 16383 pixels a side, top-down or bottom-up, converts them to 4:2:0 YCbCr with the BT.601 matrix VP8 defines, and emits one VP8 frame per input frame for `WebmMuxer`; VP8 carries no configuration record, so `EncoderConfig::decoder_config` is empty. It codes key frames and inter frames predicted from the previous frame, choosing per macroblock between the four whole-macroblock and ten subblock intra modes and `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV` with a quarter-sample motion search, skipping a residual whose bits buy too little, adapting the token probabilities to each frame, and running the loop filter. Its reconstruction is the decoder's own, so `native_vp8_video_decoder_factory()` and libvpx decode exactly the pictures it predicts from, which a test checks byte for byte against libvpx through ffmpeg. `VideoEncoderConfig::configuration` takes the HEVC encoder's forms with a VP8 quantizer: empty is a fixed quantizer index of 32, one byte a fixed quantizer index from 0 to 127, and four big-endian bytes a target bit rate in bits a second, optionally followed by four more giving the keyframe interval in frames; without one, a key frame starts every second. Under a target bit rate the quantizer follows each frame's size towards the budget, carrying any surplus or deficit forward, so over two seconds of the tests' moving pattern the average lands within 2 percent of the target. Encoding runs on the calling thread at about 400 ms a 1080p frame on one desktop core, so it suits short clips and small frames rather than real-time HD capture. There is no hardware VP8 encoder behind it, so `Require` reports `HardwareUnavailable`.

`native_vp9_video_encoder_factory()` supplies a dependency-free software backend for VP9 profile 0 (8-bit 4:2:0), written to MP4 as a `vp09` sample entry with a `vpcC` box naming the level the frame size and rate need, or to WebM as a `V_VP9` track. It accepts `Yuv420p8`, `Rgba8`, `Bgra8`, or `Gray8` CPU frames of any size up to 4096 pixels wide, top-down or bottom-up; RGB is converted to 4:2:0 with the BT.601 matrix in the configured range, and the bitstream signals BT.601. It emits key frames and inter frames: a key frame starts every group of pictures, and each inter frame predicts from the frame before it with motion vectors from a whole-pixel diamond search refined to quarter pixels, or eighth pixels below quantizer index 200, through VP9's regular 8-tap filters, splitting each 64x64 superblock into blocks from 64x64 down to 8x8 and choosing per block between motion compensation and DC, V, H, or TM intra prediction, with transforms from 4x4 up to 32x32, all by rate and distortion. Key frames are sync samples marked independent and inter frames are marked dependent, so `ExactFrameReader` restarts a seek at the nearest key frame. Frames are deblocked by the VP9 loop filter at the level that brings them closest to the source, and adapt their entropy coding probabilities from frame to frame as the decoder does, so the coding follows the content; each key frame starts again from VP9's default probabilities. `VideoEncoderConfig::configuration` is empty (quantizer index 80 and a key frame every 60 frames), a single nonzero `base_q_idx` byte (higher is smaller and softer), that byte followed by a nonzero big-endian `u16` key frame interval in frames, or those three bytes followed by a flags byte. Flag bit 0 (`1`) codes every frame error resilient, with the default probabilities and without the previous frame's motion vectors, so each frame decodes from its reference picture alone at the cost of a larger stream; the other bits must be zero. The output decodes identically, pixel for pixel, through ffmpeg's VP9 decoder and libvpx, and plays through the browser's own `WebCodecs` VP9 decoder. It also round-trips frame-accurately through `native_vp9_video_decoder_factory()`, sequentially and under seeks across key frames.

The same factory selects a hardware VP9 encoder when `VideoEncoderConfig::hardware` asks for one. On Windows, `Prefer` and `Require` route to the GPU vendor's hardware Media Foundation VP9 encoder where one is registered, such as Intel Quick Sync's; on macOS, to VideoToolbox's hardware VP9 encoder on Macs whose media engine has one. Either takes limited-range input at even dimensions: `Rgba8`, `Bgra8`, or `Yuv420p8` on Windows, and `Rgba8` or `Bgra8` on macOS. The hardware encoder takes the same `configuration` as the software one: it is rate controlled by quality, which `base_q_idx` maps onto, with a key frame at least every configured interval. It cannot code error-resilient frames, so a configuration with the error-resilient flag always encodes in software. Its stream is muxed exactly as the software encoder's is, as `vp09`/`vpcC` in MP4 and `V_VP9` in WebM, with a `vpcC` built from the color space and range its first key frame signals, and a sample is a sync sample exactly when it opens on a key frame. `capability()` reports `Supported { implementation: Hardware }` only when such an encoder is usable for the configuration; otherwise `Prefer` falls back to the software encoder and `Require` reports `CodecSupport::HardwareUnavailable`, or `InvalidConfiguration` for a configuration no hardware encoder takes. `Avoid` always encodes in software, and Linux has no hardware VP9 encoder yet.

`native_hevc_video_encoder_factory()` supplies the matching dependency-free software backend for 8-bit 4:2:0 HEVC Main profile. It accepts limited-range `Rgba8` CPU frames whose dimensions are multiples of 16, emits one IDR access unit per input frame, and generates the matching standardized `hvcC` box. `VideoEncoderConfig::configuration` selects the operating point the same way AV1's does: empty codes every coding unit as a `pcm_flag == 1` PCM block, so the coded picture is exactly the source, a single byte carrying `SliceQpY` in `0..=51` codes ITU-T H.265 §7.3.8.11 quantized residual at that QP, and four big-endian bytes give a nonzero target bitrate in bits a second, optionally followed by four more giving a keyframe interval in frames (zero meaning one second), which the native writer meets trivially because every picture it writes is an IDR. Lossless PCM stays the default, so callers that configure nothing get byte-identical output to earlier releases; unlike AV1's `base_q_idx`, a `0` byte is not a lossless request, because HEVC QP 0 is simply the finest quantizer step. The two lossy forms differ in what they optimize as well as in who picks the quantizer: a fixed QP takes the closest picture at that QP, because a caller naming a QP is asking for a picture, while a bitrate target picks `SliceQpY` per picture against what the previous picture actually cost and charges every intra candidate for the residual bits it would code, because with a rate to hit the bits a decision saves buy quality elsewhere. Lossy streams decode through `native_hevc_video_decoder_factory()` with distortion that tracks the requested QP, and a bitrate-targeted stream settles within 25 percent of its per-picture budget — about as tight as an integer QP chosen once a picture allows. The writer searches the intra prediction mode per coding unit and runs both in-loop filters -- §8.7.2 deblocking, and §8.7.3 SAO wherever the pass earns the `sao( )` syntax it costs against the picture's own rate-distortion curve. `VideoEncoderConfig::timescale` and `frame_duration` define the exact constant-rate clock used for emitted DTS, PTS, and duration values.

The same factory selects a platform encoder when `VideoEncoderConfig::hardware` asks for one. On Windows, `Prefer` and `Require` route a target-bitrate configuration to Media Foundation: the GPU vendor's asynchronous hardware HEVC encoder (NVENC, Quick Sync, or AMF) first, and under `Prefer` Microsoft's software HEVC encoder from the HEVC Video Extensions next and the native encoder last; `Require` reports `CodecSupport::HardwareUnavailable` instead of falling back, and a lossless or fixed-QP configuration, which has no hardware form, is an invalid configuration for it. The Media Foundation backend encodes HEVC Main at a constant frame rate with the requested bitrate and keyframe interval, no B-frames (so every sample's DTS equals its PTS), and low latency. It accepts limited-range `Rgba8`, `Bgra8`, or `Yuv420p8` CPU frames with even dimensions: RGB goes to a hardware encoder that takes ARGB32 as-is, so the color conversion runs on the GPU, and otherwise converts to NV12 on the CPU, in both cases with the BT.601 studio-swing matrix the crate's decoder uses. Its Annex B output is rewritten as length-prefixed `hvc1` samples, and the `hvcC` is built from the parameter sets the encoder actually wrote, so the stream muxes through `Mp4Muxer` and decodes through `native_hevc_video_decoder_factory()` and ffmpeg alike. `finish()` drains the encoder; dropping a pending `encode()` or `finish()` future cancels the stream, after which every call reports `ErrorKind::Canceled`; and a lost GPU device is reported as `ErrorKind::Graphics`, after which a new encoder must be created. `VideoEncoder::implementation()` and `VideoEncoder::backend_name()` say which encoder a factory actually created -- the latter names the Media Foundation encoder, for example `NVIDIA HEVC Encoder MFT`. Linux has no platform HEVC encoder yet, so there `Prefer` selects the native encoder and `Require` is unavailable.

On macOS, `Prefer` and `Require` route a target-bitrate configuration to VideoToolbox's hardware HEVC encoder, through a `VTCompressionSession` created with hardware required; `Prefer` falls back to the native encoder where the host has none, and `Require` reports `CodecSupport::HardwareUnavailable`. It encodes HEVC Main in real-time mode at a constant frame rate with the requested bitrate and keyframe interval and no B-frames, so every sample's DTS equals its PTS. It accepts limited-range `Rgba8` or `Bgra8` CPU frames with even dimensions, top-down or bottom-up: the pixels are copied into BGRA pixel buffers - as they are for `Bgra8`, with the channels swapped for `Rgba8` - and VideoToolbox converts them to YCbCr on the media engine with the BT.601 matrix, so no CPU color conversion runs. Because `config()` has to declare the `hvcC` before the first frame and VideoToolbox only reveals its parameter sets once it has encoded something, creating the encoder encodes one black priming frame first and discards its sample. VideoToolbox's samples are already length-prefixed; the parameter sets are moved out of them into the `hvcC`, and sync flags come from the encoder. In real time VideoToolbox may drop a frame it cannot keep up with, and the sample before a dropped frame is lengthened to cover it, so decode timestamps stay contiguous. `finish()` drains the encoder, and dropping it unfinished cancels it. `backend_name()` reports `VideoToolbox HEVC`. On `macos-latest` CI it encodes 1080p at about 180 fps, six times real time.

Frame indices are zero-based. A video `get(n)` returns exactly frame `n` in presentation order, not merely the nearest keyframe. The matching audio `get(n)` returns the half-open sample interval covered by video frame `n`; audio-only callers read exact sample ranges instead, with `AudioSampleReader` natively and `AudioStream.getRange()` in the browser.

### A.4 Native decoders

The HEVC decoder also takes HEVC Main 10 (`CodecProfile::HevcMain10`, 8- to 10-bit 4:2:0) on every platform, always in pure Rust: none of the hardware backends is wired for more than 8 bits per sample, so `Require` reports a Main 10 track unavailable.

`native_vp8_video_decoder_factory` selects its backends the same way: NVDEC on 64-bit Windows and Linux hosts with an NVIDIA adapter that decodes VP8, then Media Foundation on Windows hosts whose adapter exposes the D3D11 VP8 decoder profile (Intel and some AMD drivers do; NVIDIA drivers do not) and that have a D3D11-aware VP8 decoder transform installed, such as the one in Microsoft's VP9 Video Extensions, then pure Rust, and pure Rust on macOS, where VideoToolbox has no VP8 decoder. The hardware backends' frames are cropped and converted to RGBA by the same code as the pure-Rust decoder's, so every backend returns the same pixels.

`native_vp9_video_decoder_factory()` decodes VP9 profile 0 (`CodecProfile::Vp9Profile0`, 8-bit 4:2:0) to `Rgba8`. `Prefer` and `Require` select a hardware backend: NVDEC on 64-bit Windows and Linux hosts with an NVIDIA adapter that decodes VP9, then Media Foundation on Windows hosts whose adapter exposes the D3D11 VP9 profile 0 decoder profile and that have a D3D11-aware VP9 decoder transform installed, such as the one in Microsoft's VP9 Video Extensions, then VideoToolbox on Macs whose media engine decodes VP9. `Prefer` falls back to pure Rust where none is available, `Require` reports `HardwareUnavailable` there, and `Avoid` always decodes in pure Rust. A chunk's headers are read before it reaches the hardware decoder, so every backend refuses the samples the pure-Rust decoder refuses and returns exactly one frame per sample, and the decoded picture is cropped to the size its header names and converted by the pure-Rust decoder's own conversion, so every backend returns the same pixels. NVDEC's parser never displays a `show_existing_frame`, so that backend reads the surface the named reference slot holds back itself. VideoToolbox never outputs a hidden frame, so that backend shows a hidden frame again by replaying the samples since the last key frame through the pure-Rust decoder, which encoders make it do rarely. The hardware backends decode a stream at its configured size only. It reads VP9 from MP4 `vp09` sample entries, whose `vpcC` box is the track's `decoder_config`, and from WebM `V_VP9` tracks, whose `CodecPrivate` becomes the same box (`Vp9CodecConfig` parses both), and takes each sample as one VP9 chunk: a frame, or a superframe of hidden frames followed by the one it shows. Every coding tool of the profile is decoded, including `show_existing_frame`, intra-only frames, reference frames of another size, tiles, segmentation and lossless coding, and the output matches libvpx bit for bit on every profile 0 test vector of libvpx's own test suite. A stream that changes resolution produces frames of each size, which `ExactFrameReader` refuses because it holds every frame to the track's configured size. Like the other software decoders, it decodes on the calling thread and suits thumbnails, previews and seeks rather than real-time playback of large frames.

The native audio decoders are pure Rust except AAC's on macOS and Windows: `NativeAacDecoder` runs the platform's AAC-LC decoder there, AudioToolbox on macOS and Media Foundation's AAC decoder MFT on Windows, and Symphonia's AAC decoder on Linux and the other native targets, which have no platform AAC decoder; it takes mono and stereo streams. `NativeVorbisDecoder` runs a copy of Symphonia's Vorbis decoder vendored in `crates/zvidlib-vorbis-decoder/src/vorbis_decoder/` with its surround decoding fixed, for one to eight channels in the Vorbis channel order, and `NativeOpusDecoder` runs `opus-pure`, which passes all twelve RFC 8251 decoder conformance vectors in stereo and in mono. Opus is read from and written to MP4 (`Opus` sample entries with `dOps`) and WebM; Vorbis has no MP4 mapping, so it is read from and written to WebM, `VorbisConfig` parses and writes the Xiph-laced `CodecPrivate` WebM carries, and `Mp4Muxer` refuses a Vorbis track.

### A.5 Conformance

Native compressed-codec backends use the public `VideoDecoderConformanceVector`
and `VideoEncoderConformanceVector` runners before registration. Decoder vectors
pin canonical SHA-256 fingerprints for every presentation frame and are tested
under sequential, reverse, and seek-heavy access. Encoder vectors validate
standard configuration and packet timing, then decode through an independently
conforming backend and enforce an explicit PSNR floor. This gives HEVC and AV1
work finite, reusable acceptance targets without delegating codec behavior to an
external library.

The VP9 decoder is additionally checked against every VP9 profile 0 test vector
libvpx's own test suite decodes, which CI downloads from the WebM project. To
run that check locally, download the vectors named in
`crates/zvidlib-vp9-decoder/tests/fixtures/libvpx_vp9_test_vectors.txt`, each with its `.md5` file,
from `https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`
into one directory, then:

```console
ZVIDLIB_VP9_VECTORS=/path/to/vectors cargo test -p zvidlib-vp9-decoder --lib vp9_dec::tests::libvpx_test_vectors -- --ignored
```
