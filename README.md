# zvidlib

zvidlib is a Rust library for frame-accurate video and synchronized audio I/O on native and WebAssembly targets. Its primary jobs are reading an MP4 into a GL/WebGL canvas and writing canvas frames plus an audio stream into an MP4, behind a small API centered on indexed `get` and `put` operations.

> **Project status:** zvidlib is pre-1.0 and its API may still change in a future minor release. The portable foundation now includes checked timeline arithmetic, validated media buffers, asynchronous byte I/O, CPU/GL/WebGL transfer contracts, bounded ordinary/fragmented MP4 sample indexing, normalized codec factories, bounded exact-frame video decoding, exact AAC, Opus and Vorbis sample reads, audio-clock playback control and adapter contracts, encoder contracts, strict indexed output, seekable MP4 muxing, and the browser WebAssembly boundary. The generated JavaScript package includes BigInt-safe values, stable errors, Blob/stream input, Blob output, session/stream/playback handles, real HEVC/AV1/VP9 video decode through the browser's native `WebCodecs` `VideoDecoder` (falling back to the pure-Rust HEVC Main/Main 10, AV1 and VP9 profile 0 decoders when `WebCodecs` cannot decode a track), and AAC packet/config exports for browser `AudioDecoder` playback. Native builds include HEVC Main and AV1 Main decoders and encoders (`native_hevc_video_decoder_factory`, `native_hevc_video_encoder_factory`, `native_av1_video_decoder_factory`, `native_av1_video_encoder_factory`), a VP9 profile 0 decoder and encoder (`native_vp9_video_decoder_factory`, `native_vp9_video_encoder_factory`), plus AAC-LC decode and default-device PCM output for synchronized playback; HEVC decode uses NVIDIA NVDEC on supported 64-bit Windows/Linux systems, a D3D11-aware Media Foundation decoder on Windows, or VideoToolbox on macOS, and otherwise retains the dependency-free pure-Rust fallback; HEVC encode can use a hardware Media Foundation encoder (NVENC, Quick Sync, or AMF) on Windows or VideoToolbox's hardware encoder on macOS. Color AV1 encoding beyond the monochrome profile and fully portable audio-device abstractions remain planned; a pure-Rust AAC encoder is not -- see the writer-core notes under [Planned API examples](#planned-api-examples) for where each audio encoder comes from, and [Implemented browser boundary](#implemented-browser-boundary) for the browser's own `WebCodecs`-backed AAC-LC and Opus encode bridge. The complete workflow interfaces below remain intentionally aspirational.

## Documentation

Full API documentation is published at [lsegal.github.io/zvidlib](https://lsegal.github.io/zvidlib/) and rebuilt from `main` on every push.

## Goals

- Return exactly the requested video frame, including across codecs that use bidirectionally predicted frames.
- Keep sequential access fast by retaining decoded frames and codec state between requests.
- Keep audio aligned to the video timeline for indexed reads, writes, seeking, and playback.
- Move frames through CPU buffers or GL/WebGL textures and framebuffers, using zero-copy paths when the platform permits.
- Read and write ISO Base Media File Format/MP4, initially targeting HEVC/H.265, AV1 and VP9 video (VP9 written as `vp09`) with AAC or Opus audio, and WebM with Opus or Vorbis audio alongside its video.
- Present equivalent Rust and JavaScript concepts on native and `wasm32` targets.
- Make containers, codecs, storage, and graphics backends replaceable without exposing backend details in the common API.

The project is a library, not a media CLI or an FFmpeg binding. FFmpeg is a feature-set reference only; zvidlib will not copy or incorporate FFmpeg source.

## Implemented browser boundary

The `web` feature exposes `MediaInput`, `MediaOutput`, `Playback`, `VideoStream`, `AudioStream`, `OpenOptions`, `CreateOptions`, `PlaybackOptions`, `PreviewIndex`, `PreviewOptions`, `Preview`, `FrameIndex`, `Timestamp`, `Rational`, `SampleRange`, `VideoFrame`, and `AudioBuffer` through `wasm-bindgen`.

`MediaInput.open` accepts a `Blob`, `ReadableStream<Uint8Array>`, `ArrayBuffer`, or typed-array view. It consumes streams, always releases its reader lock, and supports cancellation through `OpenOptions.signal`. Input bytes are copied into owned WebAssembly storage; `bytes()` returns a fresh JavaScript snapshot rather than a view into growable WebAssembly memory.

```js
import init, { FrameIndex, MediaInput, OpenOptions, errorCode } from "./pkg/zvidlib.js";

await init();

const controller = new AbortController();
const options = new OpenOptions(64n * 1024n * 1024n);
options.signal = controller.signal;

const response = await fetch("./clip.mp4");
const input = await MediaInput.open(await response.blob(), options);
console.log(input.byteLength); // BigInt
console.log(new FrameIndex(18_446_744_073_709_551_615n).value);

try {
  // Decodes the real frame via the browser's native WebCodecs `VideoDecoder`
  // for HEVC/AV1/VP9 input tracks, or via zvidlib's own software decoder when
  // WebCodecs cannot decode the track.
  const frame = await input.video(0).get(0n);
  console.log(frame.width, frame.height, frame.pixels.length);
} catch (error) {
  // "UNSUPPORTED" if neither WebCodecs nor the software decoder can decode
  // the track, rather than a fake or nearest frame.
  console.log(errorCode(error));
}

input.close();
```

An input `AudioStream` reads exact sample ranges: `getRange(start, end)` resolves to an `AudioBuffer` holding exactly the half-open range `[start, end)` of the track's presentation samples, after its priming, end padding and edit list, and `sampleCount()` is one past the last of them. AAC and Opus tracks decode through the browser's `WebCodecs` `AudioDecoder` where it supports them, and Opus falls back to zvidlib's own decoder where it does not or where the browser's output does not line up with the track. `decoderConfig()` returns a track's `WebCodecs` `AudioDecoderConfig` for a caller driving its own decoder with `packet()`.

`MediaOutput.finish()` returns a browser-owned `Blob` with the configured MIME type. `writeEncodedChunk()` remains the raw byte-sink boundary for a caller supplying its own already-muxed container bytes. Indexed `put` on any video track encodes through a built-in `WebCodecs` bridge (AV1 Main, HEVC Main or VP9 profile 0; `CreateOptions.videoCodec` selects `"av1"` (the default), `"hevc"` or `"vp9"`) and mixes every track's samples into a real multi-track MP4 at `finish()`; call `videoEncodeSupport(hardware, codec)` first to check whether the current browser can encode the selected codec. A VP9 track is written as a `vp09` sample entry whose `vpcC` takes its colour description from the encoder's first key frame. Input frames accept `VideoFrame.rgba()`, `VideoFrame.bgra8()`, or `VideoFrame.yuv420p8()`. Track 0 of an audio stream likewise encodes through a `WebCodecs` `AudioEncoder` bridge from interleaved `f32` PCM: AAC-LC by default, or Opus with `CreateOptions.audioCodec = "opus"`, which takes 48 kHz input, the rate Opus decodes at, and writes the encoder's pre-skip and end padding into the MP4's edit list so the track reads back with exactly the samples that were put. Call `audioEncodeSupport(hardware, codec)` first; it reports `false` for `"vorbis"`, which no browser encodes and MP4 cannot carry. `finish()` muxes whichever tracks were used into one MP4. Other audio track indices and other codecs/profiles remain unsupported until their backends land. `setCoverArt(bytes, "image/jpeg" | "image/png")`, called any time before `finish()`, embeds an already-encoded picture as iTunes-style `covr` metadata, which Windows Explorer, Finder and most players show as the file's thumbnail without decoding the video. Without one, `finish()` generates the cover itself from video frame 4 of the first video track, shrunk to at most 512 pixels on its longest edge and encoded as PNG, or from the last frame of a shorter stream; set `CreateOptions.coverFrame` to another zero-based index, or to `null` for no generated cover.

```js
import { CreateOptions, MediaOutput } from "./pkg/zvidlib.js";

const createOptions = new CreateOptions("mp4");
createOptions.maxOutputBytes = 512n * 1024n * 1024n;
const output = await MediaOutput.create(createOptions);

// A future MP4 muxer supplies already-encoded container chunks here.
output.writeEncodedChunk(new Uint8Array([0, 0, 0, 8, 0x66, 0x72, 0x65, 0x65]));
const blob = await output.finish();
console.log(blob.type); // "video/mp4"
```

### WebM

WebM is read and written alongside MP4 through the same indexed `get(n)`/`put(n)` API. `MediaInput.open` tells the two apart by the bytes themselves, never by a file name or MIME type: an EBML header whose `DocType` is `webm` or `matroska` is WebM, and `input.container` reports `"webm"`, `"mp4"`, or `null`. A WebM input's AV1 video tracks then answer `get(n)`, `frameDuration(n)`, `randomAccessPoints()` and `previews()` exactly as an MP4's do. `new CreateOptions("webm")` makes `MediaOutput.finish()` mux the encoded video into a seekable WebM and return a `video/webm` Blob; `supportedContainers()` lists both names.

WebM permits only VP8, VP9 or AV1 video and Vorbis or Opus audio, so WebM output is AV1 or VP9 video with Opus or Vorbis audio: `CreateOptions.audioCodec` defaults to `"opus"` on a WebM output and refuses `"aac"`, and `setCoverArt()` and `videoCodec = "hevc"` reject with `UNSUPPORTED`. On input, Opus and Vorbis tracks read through `AudioStream.getRange()` with their `CodecDelay` and `DiscardPadding` trims applied, and other audio tracks are skipped rather than refused, so a browser `MediaRecorder` capture still opens for its video; VP8 video decodes through the browser's `WebCodecs` `vp8` decoder, or zvidlib's own VP8 decoder where `WebCodecs` cannot, and VP9 profile 0 video through `WebCodecs` with a `vp09` codec string, or zvidlib's own VP9 decoder where `WebCodecs` cannot. A VP8 frame the encoder hid (an alternate reference stored as a block of its own) is currently indexed as a frame, and asking for that frame fails. A VP9 track is written as `V_VP9`, with the profile, level, bit depth and chroma subsampling of its `vpcC` as the `CodecPrivate` features, and reads back with an equivalent `vpcC`.

```js
const options = new CreateOptions("webm");
options.setTimeline(30, 1, 48_000);
const output = await MediaOutput.create(options);
for (let frameIndex = 0n; frameIndex < 90n; frameIndex++) {
  await output.video(0).put(frameIndex, VideoFrame.rgba(width, height, pixels));
}
const webm = await output.finish(); // Blob { type: "video/webm" }
const input = await MediaInput.open(webm);
console.log(input.container); // "webm"
const frame = await input.video(0).get(45n);
```

Natively, `WebmDemuxer::open` builds the same `Mp4Track` sample index `Mp4Demuxer::open` does, so `to_encoded_video_samples` and `ExactFrameReader` read a WebM track unchanged. It handles unknown-size Segments and Clusters (as live recorders write them), SimpleBlocks and BlockGroups, Xiph, EBML and fixed-size lacing, and any `TimestampScale`; `WebmDemuxer::seek_point` starts a decode from the file's `Cues` when it has them and from the scanned keyframes when it does not. `WebmMuxer` takes the same `Mp4TrackConfig` declarations `Mp4Muxer` does, writes payload bytes as samples arrive, and at `finish` fills in the Segment and Cluster sizes, the `Duration`, a `SeekHead`, and `Cues` with one cue per keyframe, which is what lets Chrome and Firefox seek the file. Samples are written in one presentation-time order across tracks, because a WebM Cluster interleaves them. An Opus or Vorbis track is indexed into the same `Mp4Track` an MP4 audio track is - an Opus `OpusHead` rewritten as the `dOps` box `opus_config` reads, a Vorbis `CodecPrivate` as stored for `vorbis_config` - and `WebmDemuxer::audio_timing` turns its `CodecDelay` and last `DiscardPadding` into the priming and padding `AudioSampleReader` trims. `WebmMuxer` writes an Opus track's pre-skip as `CodecDelay` with an 80 ms `SeekPreRoll`, and the padding `set_audio_gapless` declares as the track's last block's `DiscardPadding`, holding each audio track's newest block back until it knows whether that block is the last; FFmpeg decodes zvidlib's Opus WebM to exactly the encoded length. An Opus or Vorbis track is indexed into the same `Mp4Track` an MP4 audio track is - an Opus `OpusHead` rewritten as the `dOps` box `opus_config` reads, a Vorbis `CodecPrivate` as stored for `vorbis_config` - and `WebmDemuxer::audio_timing` turns its `CodecDelay` and last `DiscardPadding` into the priming and padding `AudioSampleReader` trims. `WebmMuxer` writes an Opus track's pre-skip as `CodecDelay` with an 80 ms `SeekPreRoll`, and the padding `set_audio_gapless` declares as the track's last block's `DiscardPadding`, holding each audio track's newest block back until it knows whether that block is the last; FFmpeg decodes zvidlib's Opus WebM to exactly the encoded length. `probe_container` detects either container from a `ByteSource`, and `container_capabilities` reports both.

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

## Planned API examples

The examples in this section show the intended complete workflows. Rust examples are deliberately marked `ignore`, and browser examples depend on media backends that are not fully registered yet: video decoding works (see [Implemented browser boundary](#implemented-browser-boundary)), but `Playback`, audio decode/encode, and video encode still reject with `UNSUPPORTED` until their implementations land. The boundary classes exist for all of these. Names may change before the first release, while the ownership, synchronization, and cleanup steps are part of the design.

### Read, play audio, and render with native OpenGL

The application owns the window, current GL context, and audio device. zvidlib owns the demuxer and decoder state, uploads exact frames into the caller's texture, and uses the audio device clock as the playback clock.

```rust,ignore
use std::time::Duration;
use zvidlib::{AudioOutput, GlContext, OpenOptions, Playback};

#[tokio::main]
async fn main() -> zvidlib::Result<()> {
    // Application-specific setup (glutin/winit, SDL, etc.). The GL context must
    // stay current on this thread for every zvidlib and draw call below.
    let (window, gl): (_, GlContext) = create_window_and_current_gl_context()?;
    let audio_device: AudioOutput = open_default_audio_device()?;
    let texture = create_rgba_texture(&gl, window.drawable_size())?;

    let input = zvidlib::open("clip.mp4", OpenOptions::default()).await?;
    let video = input.video(0)?;
    let audio = input.audio(0)?;
    let mut playback = Playback::builder(video, audio)
        .audio_output(audio_device)
        .video_destination(gl.texture_2d(texture))
        .build()
        .await?;

    playback.play().await?;
    while !window.should_close() && !playback.is_finished() {
        window.poll_events();

        // `present` schedules audio ahead, asks `get(n)` for the exact frame
        // selected by the audio clock, and uploads that frame to `texture`.
        if playback.present().await? {
            draw_fullscreen_texture(&gl, texture)?;
            window.swap_buffers()?;
        }

        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    playback.stop().await?;
    input.close().await?;
    Ok(())
}
```

The helpers that create the native window, GL objects, shaders, and audio device depend on the application's windowing/audio crates and are intentionally left to the host. `present` never substitutes a nearby keyframe: it resolves the current presentation index and calls the same exact-frame path as `video.get(n)`.

### Read, play audio, and render with WebGL

Build the package as described in [Building and using WebAssembly](#building-and-using-webassembly), then serve this page and `clip.mp4` from the same directory. Browser security rules require HTTP(S); opening the HTML as a `file:` URL will not work reliably.

```html
<!doctype html>
<html lang="en">
  <meta charset="utf-8">
  <title>zvidlib WebGL playback</title>
  <canvas id="video" width="1280" height="720"></canvas>
  <button id="play" type="button">Play</button>
  <script type="module">
    import init, { MediaInput, Playback } from "./pkg/zvidlib.js";

    await init();

    const canvas = document.querySelector("#video");
    const playButton = document.querySelector("#play");
    const gl = canvas.getContext("webgl2", { alpha: false });
    if (!gl) throw new Error("WebGL 2 is unavailable");

    const response = await fetch("./clip.mp4");
    if (!response.ok) throw new Error(`clip.mp4: ${response.status}`);

    const input = await MediaInput.open(await response.blob());
    const video = input.video(0);
    const audio = input.audio(0);
    const audioContext = new AudioContext();
    const playback = await Playback.create({
      video,
      audio,
      audioContext,
      webgl: { context: gl, canvas },
    });

    let animationFrame = 0;
    async function render() {
      // Schedules audio and renders the exact `video.get(n)` result selected by
      // the AudioContext clock into the canvas's default framebuffer.
      const { finished } = await playback.present();
      if (!finished) animationFrame = requestAnimationFrame(render);
    }

    playButton.addEventListener("click", async () => {
      playButton.disabled = true;
      await audioContext.resume(); // Must follow a user gesture.
      await playback.play();
      animationFrame = requestAnimationFrame(render);
    }, { once: true });

    window.addEventListener("pagehide", () => {
      cancelAnimationFrame(animationFrame);
      playback.close();
      input.close();
      audioContext.close();
    }, { once: true });
  </script>
</html>
```

### Write synchronized native GL and audio input

Writing uses the same zero-based frame index. The application captures audio for the exact half-open interval represented by each video frame and submits both values before advancing the index.

```rust,ignore
use zvidlib::{CreateOptions, GlContext, VideoCodec};

#[tokio::main]
async fn main() -> zvidlib::Result<()> {
    let (window, gl): (_, GlContext) = create_window_and_current_gl_context()?;
    let microphone = open_default_audio_input()?;
    let framebuffer = create_recording_framebuffer(&gl, 1920, 1080)?;

    let mut output = zvidlib::create(
        "recording.mp4",
        CreateOptions::mp4()
            .video(VideoCodec::Av1, 1920, 1080, 30, 1)
            .aac_audio(microphone.sample_rate(), microphone.channels()),
    ).await?;
    let video = output.video(0)?;
    let audio = output.audio(0)?;

    for frame_index in 0_u64..300 {
        render_scene_into(&gl, framebuffer, frame_index)?;
        let samples = microphone.capture(audio.interval_for_frame(frame_index)?).await?;

        video.put(frame_index, gl.framebuffer(framebuffer)).await?;
        audio.put(frame_index, samples).await?;
    }

    // `finish` drains delayed video/audio packets and writes the MP4 indexes.
    output.finish().await?;
    window.close();
    Ok(())
}
```

### Write a WebGL canvas and Web Audio stream

This browser example records ten seconds. A real application can replace `drawScene` and `microphone` with any WebGL renderer and timestamped Web Audio source.

```js
import init, { MediaOutput } from "./pkg/zvidlib.js";

await init();

const canvas = document.querySelector("#video");
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
const audioContext = new AudioContext();
const microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
const output = await MediaOutput.create({
  container: "mp4",
  video: { codec: "av1", width: canvas.width, height: canvas.height, fps: 30 },
  audio: { codec: "aac", sampleRate: audioContext.sampleRate, channels: 2 },
});
const video = output.video(0);
const audio = output.audio(0);
const capture = await audio.captureFromMediaStream(audioContext, microphone);

for (let frameIndex = 0n; frameIndex < 300n; frameIndex++) {
  drawScene(gl, frameIndex);
  const samples = await capture.get(audio.intervalForFrame(frameIndex));
  await video.put(frameIndex, { webgl: gl, framebuffer: null });
  await audio.put(frameIndex, samples);
}

capture.close();
microphone.getTracks().forEach((track) => track.stop());
const mp4 = await output.finish();
const link = Object.assign(document.createElement("a"), {
  href: URL.createObjectURL(mp4),
  download: "recording.mp4",
  textContent: "Download recording",
});
document.body.append(link);
```

JavaScript frame indices are `BigInt` so the wrapper can preserve the Rust `u64` range. Both writers reject skipped or repeated indices by default, and `finish()` is required to drain codecs and finalize the MP4.

zvidlib writes **no audio codec of its own design**. Every `AudioEncoder` it ships is either the platform's or a faithful port of the codec's reference encoder, because a mediocre encoder under this crate's name is harder for callers to route around than no encoder at all, and with a port the quality to defend is the reference's. `native_aac_audio_encoder_factory()` delegates to AudioToolbox on macOS and to Media Foundation on Windows, and reports `HardwareUnavailable` elsewhere rather than falling back to an encoder of its own: a pure-Rust AAC-LC encoder means owning a filter bank, a psychoacoustic model, and rate control with no open reference to port. `native_opus_audio_encoder_factory()` runs `opus-pure`, a pure-Rust port of libopus, and `native_vorbis_audio_encoder_factory()` runs zvidlib's own port of the libvorbis 1.3.7 encoder, whose packets are bit-identical to libvorbis's; both run on every target. `MediaOutput` muxes an audio track given any `AudioEncoder`, so a caller with an encoder of its own still plugs it in.

The implemented portable writer core exposes `VideoEncoder` and `AudioEncoder` contracts plus `MediaOutput`. An encoder backend declares its MP4 codec configuration and exact output timescale, then returns `EncodedSample` values with DTS, PTS, duration, sync, and dependency metadata. `MediaOutput::put_video` accepts the shared CPU/GL/WebGL `FrameSource`; it and `put_audio` enforce zero-based consecutive indices and exact frame-aligned audio ranges. `finish` drains both encoders, records audio priming and padding in an edit list, finalizes sample indexes, and flushes the seekable `ByteSink`. `MediaOutput::set_cover_art` (or `Mp4Muxer::set_cover_art`) embeds a caller-encoded JPEG or PNG `CoverArt` in `moov/udta/meta/ilst/covr` any time before `finish`, so a recorder can pick a frame after capture; operating-system file browsers show it as the thumbnail without an HEVC or AV1 decoder, output without cover art is unchanged, and `Mp4Demuxer::cover_art` reads it back. When no cover is set, `MediaOutput` generates one: `OutputOptions::cover_source` defaults to `CoverSource::Frame(4)`, and the writer shrinks that presentation-order frame to at most 512 pixels on its longest edge as frames are written, then encodes it as a PNG at `finish` with no extra dependency. A stream with fewer frames uses its last frame, a frame supplied as a GPU resource gives no cover, and `CoverSource::None` turns the generated cover off. The capture holds one thumbnail, at most 768 KiB, and costs nothing after the chosen frame. `CoverArt::from_video_frame` makes the same thumbnail from any CPU frame. `Mp4Muxer` never generates a cover, so its output without cover art is unchanged.

`native_av1_video_encoder_factory()` supplies a dependency-free software backend for 8-bit monochrome AV1 Main profile. It accepts `Gray8` CPU frames, emits one independently decodable keyframe access unit per input frame, and generates the matching standardized `av1C` box. `VideoEncoderConfig::configuration` selects the quantization profile: empty encodes losslessly with the 4x4 WHT (the default, and the output verified against an independent ffmpeg decode), and a single byte carrying a nonzero `base_q_idx` encodes non-lossless through the forward DCT with a per-block transform-size, transform-type, and partition decision. Non-lossless streams are interchange-grade: they round-trip through `native_av1_video_decoder_factory()` and decode through an independent ffmpeg 7.1 (dav1d) within an asserted distortion bound across several quantizers, frame sizes, and content patterns. `VideoEncoderConfig::timescale` and `frame_duration` define the exact constant-rate clock used for emitted DTS, PTS, and duration values. Unsupported color formats, hardware requirements, malformed private configuration, out-of-level rates/dimensions, and resource-limit violations are rejected during capability discovery or creation.

`native_vp9_video_encoder_factory()` supplies a dependency-free software backend for VP9 profile 0 (8-bit 4:2:0), written to MP4 as a `vp09` sample entry with a `vpcC` box naming the level the frame size and rate need, or to WebM as a `V_VP9` track. It accepts `Yuv420p8`, `Rgba8`, `Bgra8`, or `Gray8` CPU frames of any size up to 4096 pixels wide, top-down or bottom-up; RGB is converted to 4:2:0 with the BT.601 matrix in the configured range, and the bitstream signals BT.601. It emits key frames and inter frames: a key frame starts every group of pictures, and each inter frame predicts from the frame before it with whole-pixel motion vectors from a diamond search, choosing per 8x8 block between motion compensation and DC, V, H, or TM intra prediction by rate and distortion. Key frames are sync samples marked independent and inter frames are marked dependent, so `ExactFrameReader` restarts a seek at the nearest key frame. Every frame is error resilient and coded with VP9's default probabilities, 4x4 transforms, and the loop filter off, which keeps the encoder small and its output plain to decode. `VideoEncoderConfig::configuration` is empty (quantizer index 80 and a key frame every 60 frames), a single nonzero `base_q_idx` byte (higher is smaller and softer), or that byte followed by a nonzero big-endian `u16` key frame interval in frames. The output decodes identically, pixel for pixel, through ffmpeg's VP9 decoder and libvpx, and plays through the browser's own `WebCodecs` VP9 decoder. The backend is software-only on every platform, so `HardwarePreference::Require` reports `CodecSupport::HardwareUnavailable`. It also round-trips frame-accurately through `native_vp9_video_decoder_factory()`, sequentially and under seeks across key frames.

`native_hevc_video_encoder_factory()` supplies the matching dependency-free software backend for 8-bit 4:2:0 HEVC Main profile. It accepts limited-range `Rgba8` CPU frames whose dimensions are multiples of 16, emits one IDR access unit per input frame, and generates the matching standardized `hvcC` box. `VideoEncoderConfig::configuration` selects the operating point the same way AV1's does: empty codes every coding unit as a `pcm_flag == 1` PCM block, so the coded picture is exactly the source, a single byte carrying `SliceQpY` in `0..=51` codes ITU-T H.265 §7.3.8.11 quantized residual at that QP, and four big-endian bytes give a nonzero target bitrate in bits a second, optionally followed by four more giving a keyframe interval in frames (zero meaning one second), which the native writer meets trivially because every picture it writes is an IDR. Lossless PCM stays the default, so callers that configure nothing get byte-identical output to earlier releases; unlike AV1's `base_q_idx`, a `0` byte is not a lossless request, because HEVC QP 0 is simply the finest quantizer step. The two lossy forms differ in what they optimize as well as in who picks the quantizer: a fixed QP takes the closest picture at that QP, because a caller naming a QP is asking for a picture, while a bitrate target picks `SliceQpY` per picture against what the previous picture actually cost and charges every intra candidate for the residual bits it would code, because with a rate to hit the bits a decision saves buy quality elsewhere. Lossy streams decode through `native_hevc_video_decoder_factory()` with distortion that tracks the requested QP, and a bitrate-targeted stream settles within 25 percent of its per-picture budget — about as tight as an integer QP chosen once a picture allows. The writer searches the intra prediction mode per coding unit and runs both in-loop filters -- §8.7.2 deblocking, and §8.7.3 SAO wherever the pass earns the `sao( )` syntax it costs against the picture's own rate-distortion curve. `VideoEncoderConfig::timescale` and `frame_duration` define the exact constant-rate clock used for emitted DTS, PTS, and duration values.

The same factory selects a platform encoder when `VideoEncoderConfig::hardware` asks for one. On Windows, `Prefer` and `Require` route a target-bitrate configuration to Media Foundation: the GPU vendor's asynchronous hardware HEVC encoder (NVENC, Quick Sync, or AMF) first, and under `Prefer` Microsoft's software HEVC encoder from the HEVC Video Extensions next and the native encoder last; `Require` reports `CodecSupport::HardwareUnavailable` instead of falling back, and a lossless or fixed-QP configuration, which has no hardware form, is an invalid configuration for it. The Media Foundation backend encodes HEVC Main at a constant frame rate with the requested bitrate and keyframe interval, no B-frames (so every sample's DTS equals its PTS), and low latency. It accepts limited-range `Rgba8`, `Bgra8`, or `Yuv420p8` CPU frames with even dimensions: RGB goes to a hardware encoder that takes ARGB32 as-is, so the colour conversion runs on the GPU, and otherwise converts to NV12 on the CPU, in both cases with the BT.601 studio-swing matrix the crate's decoder uses. Its Annex B output is rewritten as length-prefixed `hvc1` samples, and the `hvcC` is built from the parameter sets the encoder actually wrote, so the stream muxes through `Mp4Muxer` and decodes through `native_hevc_video_decoder_factory()` and ffmpeg alike. `finish()` drains the encoder; dropping a pending `encode()` or `finish()` future cancels the stream, after which every call reports `ErrorKind::Cancelled`; and a lost GPU device is reported as `ErrorKind::Graphics`, after which a new encoder must be created. `VideoEncoder::implementation()` and `VideoEncoder::backend_name()` say which encoder a factory actually created -- the latter names the Media Foundation encoder, for example `NVIDIA HEVC Encoder MFT`. Linux has no platform HEVC encoder yet, so there `Prefer` selects the native encoder and `Require` is unavailable.

On macOS, `Prefer` and `Require` route a target-bitrate configuration to VideoToolbox's hardware HEVC encoder, through a `VTCompressionSession` created with hardware required; `Prefer` falls back to the native encoder where the host has none, and `Require` reports `CodecSupport::HardwareUnavailable`. It encodes HEVC Main in real-time mode at a constant frame rate with the requested bitrate and keyframe interval and no B-frames, so every sample's DTS equals its PTS. It accepts limited-range `Rgba8` or `Bgra8` CPU frames with even dimensions, top-down or bottom-up: the pixels are copied into BGRA pixel buffers - as they are for `Bgra8`, with the channels swapped for `Rgba8` - and VideoToolbox converts them to YCbCr on the media engine with the BT.601 matrix, so no CPU colour conversion runs. Because `config()` has to declare the `hvcC` before the first frame and VideoToolbox only reveals its parameter sets once it has encoded something, creating the encoder encodes one black priming frame first and discards its sample. VideoToolbox's samples are already length-prefixed; the parameter sets are moved out of them into the `hvcC`, and sync flags come from the encoder. In real time VideoToolbox may drop a frame it cannot keep up with, and the sample before a dropped frame is lengthened to cover it, so decode timestamps stay contiguous. `finish()` drains the encoder, and dropping it unfinished cancels it. `backend_name()` reports `VideoToolbox HEVC`. On `macos-latest` CI it encodes 1080p at about 180 fps, six times real time.

Frame indices are zero-based. A video `get(n)` returns exactly frame `n` in presentation order, not merely the nearest keyframe. The matching audio `get(n)` returns the half-open sample interval covered by video frame `n`; sample-accurate APIs will also be available for audio-only use.

Runnable versions of the native GL and browser WebGL workflows above, driven against a real Big Buck Bunny MP4 sample, live in [`examples/`](examples/README.md).

## Data paths

Video frames will support two families of destination/source:

- CPU-backed planes with explicit pixel format, dimensions, stride, color space, and transfer metadata.
- GPU-backed images associated with a GL texture or framebuffer on native targets and a WebGL texture or framebuffer in browsers.

GPU interop is capability-based. A backend may share an image without a copy, copy on the GPU, or fall back through CPU memory. Callers can require a particular behavior and receive an unsupported-capability error instead of an implicit expensive fallback.

Audio buffers carry their sample format, channel layout, sample rate, exact timeline interval, and any priming or padding information needed for gapless synchronization.

## Platform expectations

| Capability | Native | WebAssembly/browser |
| --- | --- | --- |
| Input/output | Files, memory, caller storage | `Blob`, streams, memory, File System Access handles when supplied |
| Video acceleration | Pluggable software or platform backend | WebCodecs when available, otherwise the pure-Rust software decoder |
| Graphics | OpenGL-family context supplied by caller | WebGL context supplied by caller |
| Audio | Raw buffers and pluggable device integration | Web Audio buffers/nodes supplied by caller |
| Concurrency | Worker threads where safe | Async tasks; workers/threads only when browser isolation permits |

Codec and hardware availability varies by browser and operating system. Opening media will report capabilities explicitly; container support does not imply that every platform can encode or decode every advertised codec.

The native HEVC Main backends each factory can select, in the order it tries them. `Avoid` always takes the pure-Rust backend; `Require` takes only the hardware ones.

| HEVC Main | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Decode (`native_hevc_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | VideoToolbox, then pure Rust | NVDEC (64-bit), then pure Rust |
| Encode (`native_hevc_video_encoder_factory`) | Media Foundation hardware (NVENC, Quick Sync, AMF), then Media Foundation software, then pure Rust | VideoToolbox hardware, then pure Rust | pure Rust |

The decoder also takes HEVC Main 10 (`CodecProfile::HevcMain10`, 8- to 10-bit 4:2:0) on every platform, always in pure Rust: none of the hardware backends is wired for more than 8 bits per sample, so `Require` reports a Main 10 track unavailable.

The VP8 and VP9 decoders choose the same way:

| Decode | Windows | macOS | Linux |
| --- | --- | --- | --- |
| VP8 (`native_vp8_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | pure Rust | NVDEC (64-bit), then pure Rust |
| VP9 profile 0 (`native_vp9_video_decoder_factory`) | NVDEC (64-bit), Media Foundation (D3D11), then pure Rust | VideoToolbox, then pure Rust | NVDEC (64-bit), then pure Rust |

`native_vp8_video_decoder_factory` selects its backends the same way: NVDEC on 64-bit Windows and Linux hosts with an NVIDIA adapter that decodes VP8, then Media Foundation on Windows hosts whose adapter exposes the D3D11 VP8 decoder profile (Intel and some AMD drivers do; NVIDIA drivers do not) and that have a D3D11-aware VP8 decoder transform installed, such as the one in Microsoft's VP9 Video Extensions, then pure Rust, and pure Rust on macOS, where VideoToolbox has no VP8 decoder. The hardware backends' frames are cropped and converted to RGBA by the same code as the pure-Rust decoder's, so every backend returns the same pixels.

`native_vp9_video_decoder_factory()` decodes VP9 profile 0 (`CodecProfile::Vp9Profile0`, 8-bit 4:2:0) to `Rgba8`. `Prefer` and `Require` select a hardware backend: NVDEC on 64-bit Windows and Linux hosts with an NVIDIA adapter that decodes VP9, then Media Foundation on Windows hosts whose adapter exposes the D3D11 VP9 profile 0 decoder profile and that have a D3D11-aware VP9 decoder transform installed, such as the one in Microsoft's VP9 Video Extensions, then VideoToolbox on Macs whose media engine decodes VP9. `Prefer` falls back to pure Rust where none is available, `Require` reports `HardwareUnavailable` there, and `Avoid` always decodes in pure Rust. A chunk's headers are read before it reaches the hardware decoder, so every backend refuses the samples the pure-Rust decoder refuses and returns exactly one frame per sample, and the decoded picture is cropped to the size its header names and converted by the pure-Rust decoder's own conversion, so every backend returns the same pixels. NVDEC's parser never displays a `show_existing_frame`, so that backend reads the surface the named reference slot holds back itself. VideoToolbox never outputs a hidden frame, so that backend shows a hidden frame again by replaying the samples since the last key frame through the pure-Rust decoder, which encoders make it do rarely. The hardware backends decode a stream at its configured size only. It reads VP9 from MP4 `vp09` sample entries, whose `vpcC` box is the track's `decoder_config`, and from WebM `V_VP9` tracks, whose `CodecPrivate` becomes the same box (`Vp9CodecConfig` parses both), and takes each sample as one VP9 chunk: a frame, or a superframe of hidden frames followed by the one it shows. Every coding tool of the profile is decoded, including `show_existing_frame`, intra-only frames, reference frames of another size, tiles, segmentation and lossless coding, and the output matches libvpx bit for bit on every profile 0 test vector of libvpx's own test suite. A stream that changes resolution produces frames of each size, which `ExactFrameReader` refuses because it holds every frame to the track's configured size. Like the other software decoders, it decodes on the calling thread and suits thumbnails, previews and seeks rather than real-time playback of large frames.

Every native encoder, by operating system, and what does the encoding. A *hardware* encoder runs on the GPU or media engine. A *platform* encoder is the operating system's own codec; it reports `CodecImplementation::Hardware` because zvidlib runs none of the codec itself. A *software* encoder is zvidlib's own pure-Rust code and reports `CodecImplementation::Software`. `VideoEncoder::implementation()` and `backend_name()` report which one a created video encoder is.

| Encoder | Windows | macOS | Linux |
| --- | --- | --- | --- |
| HEVC Main (`native_hevc_video_encoder_factory`) | **Hardware**: Media Foundation (NVENC, Quick Sync, AMF) for `Prefer`/`Require` target-bitrate configurations, with `Prefer` falling back to Media Foundation's **software** HEVC encoder. **Software**: pure Rust for `Avoid`, lossless, fixed QP, and the last `Prefer` fallback | **Hardware**: VideoToolbox for `Prefer`/`Require` target-bitrate configurations (`Rgba8`/`Bgra8`, even dimensions). **Software**: pure Rust for `Avoid`, lossless, fixed QP, and the `Prefer` fallback | **Software**: pure Rust; `Require` reports `HardwareUnavailable` |
| AV1 Main, 8-bit monochrome (`native_av1_video_encoder_factory`) | **Software**: pure Rust | **Software**: pure Rust | **Software**: pure Rust |
| VP9 profile 0 (`native_vp9_video_encoder_factory`) | **Software**: pure Rust; `Require` reports `HardwareUnavailable` | **Software**: pure Rust; `Require` reports `HardwareUnavailable` | **Software**: pure Rust; `Require` reports `HardwareUnavailable` |
| AAC-LC (`native_aac_audio_encoder_factory`) | **Platform**: Media Foundation AAC encoder MFT (44.1 or 48 kHz, mono or stereo) | **Platform**: AudioToolbox | None: reports `HardwareUnavailable` |
| Opus (`native_opus_audio_encoder_factory`) | **Software**: `opus-pure`, a pure-Rust port of libopus (48 kHz, mono or stereo) | **Software**: `opus-pure` | **Software**: `opus-pure` |
| Vorbis (`native_vorbis_audio_encoder_factory`) | **Software**: zvidlib's port of the libvorbis 1.3.7 encoder (mono or stereo, every rate libvorbis supports) | **Software**: the libvorbis port | **Software**: the libvorbis port |

| Audio decoder | Windows | macOS | Linux |
| --- | --- | --- | --- |
| AAC-LC (`NativeAacDecoder`) | **Platform**: Media Foundation AAC decoder MFT (mono or stereo) | **Platform**: AudioToolbox (mono or stereo) | None: `NativeAacDecoder::new` reports `ErrorKind::Unsupported` |
| Opus (`NativeOpusDecoder`) | **Software**: `opus-pure` | **Software**: `opus-pure` | **Software**: `opus-pure` |
| Vorbis (`NativeVorbisDecoder`) | **Software**: vendored Symphonia Vorbis decoder | **Software**: vendored Symphonia Vorbis decoder | **Software**: vendored Symphonia Vorbis decoder |

The native audio decoders are pure Rust on every platform except AAC's: `NativeAacDecoder` runs the platform's AAC-LC decoder, AudioToolbox on macOS and Media Foundation's AAC decoder MFT on Windows, for mono and stereo streams, and reports `ErrorKind::Unsupported` on Linux and other targets, which have no platform AAC decoder. `NativeVorbisDecoder` runs a copy of Symphonia's Vorbis decoder vendored in `src/vorbis_decoder/` with its surround decoding fixed, for one to eight channels in the Vorbis channel order, and `NativeOpusDecoder` runs `opus-pure`, which passes all twelve RFC 8251 decoder conformance vectors in stereo and in mono. Opus is read from and written to MP4 (`Opus` sample entries with `dOps`) and WebM; Vorbis has no MP4 mapping, so it is read from and written to WebM, `VorbisConfig` parses and writes the Xiph-laced `CodecPrivate` WebM carries, and `Mp4Muxer` refuses a Vorbis track.

The browser build encodes through `WebCodecs` instead: AV1 Main, HEVC Main or VP9 profile 0 video and AAC-LC or Opus audio, wherever the browser provides those encoders (see [Implemented browser boundary](#implemented-browser-boundary)).

## Repository layout

The repository contains a dependency-free portable core that validates native and WASM build configuration and implements the foundational timeline, media, I/O, frame-transfer, bounded MP4 reading, codec factory, exact-frame decoding, encoder, indexed-output, and seekable MP4 writing layers. Planned modules and dependency boundaries are described in [ARCHITECTURE.md](ARCHITECTURE.md). Runtime dependencies will be added only with a documented portability, maintenance, size, and licensing rationale.

## Using a release

Each GitHub release contains a Rust crate archive and a browser package. For zvidlib 0.3.0, use
the release tag as the native Cargo dependency coordinate:

```toml
[dependencies]
zvidlib = { git = "https://github.com/lsegal/zvidlib.git", tag = "v0.3.0", features = ["native"] }
```

Cargo fetches the versioned release tag itself; applications do not need to clone or vendor zvidlib.
The matching `zvidlib-0.3.0.crate` GitHub Release asset provides an archive for inspection or
offline packaging. The project is not currently published to crates.io.
The native API requires Rust 1.85 or later, as recorded by `rust-version` in `Cargo.toml`; platform codec
adapters retain their documented platform capability checks.

For a browser build, install the matching release asset directly:

```sh
npm install https://github.com/lsegal/zvidlib/releases/download/v0.3.0/zvidlib-web-v0.3.0.tgz
```

The package contains `zvidlib.js`, `zvidlib.d.ts`, and `zvidlib_bg.wasm` generated by
`wasm-pack --target web`; import it as an ES module and initialize it once as shown below. Browser
and Cargo consumers must use the same release version when they exchange media or API contracts.

zvidlib follows pre-1.0 semantic versioning: patch releases preserve documented public API and
behavior, while a minor release may make breaking API changes. Every release tag has matching
Rust crate and browser package assets; release notes call out any platform
capability change.

## Building the library

Install stable Rust with the `wasm32-unknown-unknown` target, then run:

```console
cargo check --features native
cargo check --target wasm32-unknown-unknown --no-default-features --features web
cargo fmt --all -- --check
cargo clippy --all-targets --features native -- -D warnings
```

These commands validate the portable core on both targets. Native builds include accelerated HEVC Main decode through NVDEC on supported 64-bit Windows/Linux systems, Media Foundation on Windows, and VideoToolbox on macOS, with a dependency-free pure-Rust fallback. They also include the pure-Rust AV1 Main decoder (8-bit 4:2:0 colour and monochrome streams with every Main-profile coding tool and in-loop filter, verified frame-for-frame against an independent libdav1d decode), the pure-Rust VP9 profile 0 decoder (verified against libvpx's per-frame digests of its own test vectors), a dependency-free VP9 profile 0 encoder with inter frames (verified pixel for pixel against ffmpeg's VP9 decoder and libvpx), a VP8 decoder (`native_vp8_video_decoder_factory`, every bitstream version, bit-exact with libvpx on its VP8 test vectors, including hidden alternate-reference frames) that uses NVDEC or Media Foundation where available and pure Rust otherwise, dependency-free HEVC Main and monochrome AV1 Main-profile encoders (lossless, plus a non-lossless path verified against both the crate's own decoder and an independent ffmpeg decode), AAC-LC, Opus and Vorbis decode, pure-Rust Opus and Vorbis encoders, and default-device PCM output described above. Native builds also encode HEVC Main in hardware through VideoToolbox on macOS and Media Foundation on Windows, and AAC-LC through AudioToolbox on macOS and Media Foundation on Windows. The encoder table under [Platform expectations](#platform-expectations) lists every encoder by operating system. The browser build reads exact AAC, Opus and Vorbis sample ranges through `WebCodecs` `AudioDecoder` with software Opus and Vorbis fallbacks, and encodes AAC-LC, Opus or Vorbis through its own bridge (see [Implemented browser boundary](#implemented-browser-boundary)). Color AV1 encoding beyond the monochrome profile and fully portable audio-device abstractions remain planned.

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
`tests/fixtures/codec/libvpx_vp9_test_vectors.txt`, each with its `.md5` file,
from `https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/`
into one directory, then:

```console
ZVIDLIB_VP9_VECTORS=/path/to/vectors cargo test --features native --lib vp9_dec::tests::libvpx_test_vectors -- --ignored
```

## Building, testing, and using WebAssembly

The `web` feature excludes native-only integrations. During scaffold development, install the Rust target and verify it directly:

```console
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --no-default-features --features web
```

Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) and create a browser-ready ES module package:

```console
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --no-default-features --features web
python -m http.server 8000
```

The build creates `pkg/zvidlib.js`, `pkg/zvidlib.d.ts`, `pkg/zvidlib_bg.wasm`, and package metadata. Open `http://localhost:8000/` and import the generated module with `import init, { ... } from "./pkg/zvidlib.js"`, as in the browser examples. Call `await init()` exactly once before using an exported class. Deploy the generated JavaScript and WASM files together, with the server returning `application/wasm` for the `.wasm` file.

For the browser example specifically, `examples/web_canvas/package.json` wraps the `wasm-pack build` and serving steps above into one `pnpm dev` command; see [examples/README.md](examples/README.md#web-canvas-web_canvas).

Run the browser integration suite in an installed Chrome browser with:

```console
wasm-pack test --headless --chrome --no-default-features --features web
```

The suite verifies Blob and stream input, cancellation, reader-lock cleanup, BigInt range handling, typed-array copy lifetimes, browser-object ownership, stable errors, Blob output, decoding the bundled HEVC sample and a VP9 track through WebCodecs, and the software fallback for browsers whose WebCodecs cannot decode a track. The base build does not require WASM threads or cross-origin isolation. Future optional threaded builds will document their additional headers and browser requirements separately.

The `web` feature's video decoder uses `web-sys`'s `WebCodecs` bindings, which that crate gates behind `--cfg=web_sys_unstable_apis` because the spec is still evolving. `.cargo/config.toml` sets that flag for the `wasm32-unknown-unknown` target automatically, so the `cargo`/`wasm-pack` commands above need no extra flags.

## Roadmap

1. Specify and test timeline arithmetic, frame/audio value types, errors, and capability discovery.
2. Build incremental MP4 parsing and writing with deterministic fixture tests.
3. Add decoder/encoder traits and one end-to-end decode path before expanding codec coverage.
4. Add native GL and WebGL transfer backends, then synchronized playback and recording adapters.
5. Stabilize Rust and JavaScript APIs after cross-platform conformance testing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, seeking and caching behavior, MP4 responsibilities, and portability constraints.
