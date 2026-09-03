// See examples/README.md for setup: build the wasm package into ./pkg before serving this
// directory over HTTP(S). BigBuckBunny.mp4 is already symlinked in from examples/media/.
import init, {
  MediaInput,
  PreviewOptions,
  errorCode,
  seekLatencyBudgetMs,
} from "./pkg/zvidlib.js";
// The walk's pure decisions live apart from the browser wiring so `scrub.test.js` can assert
// them, the way `examples/native_gl/scrub.rs` asserts its own copy of the same rules.
import { scrubWalkStart, shouldDrawPreview } from "./scrub.js";

const canvas = document.querySelector("#video");
const playButton = document.querySelector("#play");
const rewindButton = document.querySelector("#rewind");
const fastForwardButton = document.querySelector("#fast-forward");
const previousFrameButton = document.querySelector("#previous-frame");
const nextFrameButton = document.querySelector("#next-frame");
const timeline = document.querySelector("#timeline");
const status = document.querySelector("#status");
const fps = document.querySelector("#fps");
const seekLatency = document.querySelector("#seek-latency");

const VERTEX_SHADER = `
  attribute vec2 position;
  varying vec2 uv;
  void main() {
    uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  precision mediump float;
  varying vec2 uv;
  uniform sampler2D frame;
  void main() {
    gl_FragColor = texture2D(frame, vec2(uv.x, 1.0 - uv.y));
  }
`;

function compileProgram(gl) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

// Builds a synthetic RGBA gradient frame, used only as a fallback when the browser cannot
// decode the sample's HEVC track via WebCodecs (see examples/README.md).
function syntheticFrame(width, height, phase) {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 4;
      pixels[offset] = Math.floor((255 * x) / width);
      pixels[offset + 1] = Math.floor((255 * y) / height);
      pixels[offset + 2] = Math.floor(255 * phase);
      pixels[offset + 3] = 255;
    }
  }
  return pixels;
}

async function main() {
  await init();

  const gl = canvas.getContext("webgl2", { alpha: false });
  if (!gl) throw new Error("WebGL 2 is unavailable");

  const program = compileProgram(gl);
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const positionLocation = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  status.textContent = "Fetching BigBuckBunny.mp4…";
  const response = await fetch("./BigBuckBunny.mp4");
  if (!response.ok) {
    status.textContent = `BigBuckBunny.mp4: HTTP ${response.status}. See examples/README.md for how to fetch the sample.`;
    return;
  }

  const input = await MediaInput.open(await response.blob());
  const video = input.video(0);
  const audio = input.audio(0);
  const lines = [
    `Opened ${input.byteLength} bytes.`,
    `Video stream 0 direction: ${video.direction}`,
    `Audio stream 0 direction: ${audio.direction}`,
  ];

  // Real decoding depends on the browser having an HEVC WebCodecs decoder; fall back to a
  // synthetic gradient sized to the track when it doesn't (see examples/README.md).
  let decodedFrame = null;
  try {
    decodedFrame = await video.get(0n);
    lines.push(`video.get(0n) decoded a real ${decodedFrame.width}x${decodedFrame.height} frame.`);
  } catch (error) {
    lines.push(`video.get(0n) rejected with ${errorCode(error)}: falling back to a synthetic frame.`);
  }
  const useRealDecode = decodedFrame !== null;
  let audioState = null;
  try {
    audioState = await prepareAudio(audio);
    lines.push(`AAC audio ready: ${audioState.config.sampleRate} Hz, ${audioState.config.numberOfChannels} channels.`);
  } catch (error) {
    lines.push(`AAC audio unavailable (${errorCode(error) ?? error?.name ?? "ERROR"}): playback will be silent.`);
  }
  const frameStarts = await buildFrameStarts(video);
  // Where a decode can start from, which is what a backwards scrub walks forwards from (see
  // `scrubTo`). A track this build cannot read the sync samples of scrubs from frame zero, which
  // is the only start every track has.
  let randomAccessPoints = [0];
  try {
    randomAccessPoints = (await video.randomAccessPoints()).map(Number);
    if (randomAccessPoints.length === 0) randomAccessPoints = [0];
  } catch (error) {
    lines.push(`video.randomAccessPoints() rejected with ${errorCode(error)}: scrubbing backwards restarts at frame 0.`);
  }
  const mediaDurationMs = frameStarts[frameStarts.length - 1] || 1;
  const lastFrameIndex = Math.max(0, frameStarts.length - 2);
  timeline.max = String(lastFrameIndex);
  status.textContent = lines.join("\n");

  let playing = false;
  let frameIndex = 0;
  let stopped = false;
  let videoClockStart = 0;
  let pausedMediaTimeMs = 0;

  function uploadFrame(pixels, width, height) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // The seek preview tier (issue #432). `ARCHITECTURE.md` section 3.2 requires a seek to any
  // position of any track to answer inside `seekLatencyBudgetMs()`, and the bundled sample codes
  // its 768 frames as one group of pictures, so walking to an arbitrary position is seconds of
  // decode - which is what this example used to do on every drag. The index keeps one shrunk
  // picture every stride frames, and a drag draws the nearest of them at once while the walk
  // goes after the exact frame underneath it.
  const budgetMs = seekLatencyBudgetMs();
  let previews = null;
  let worstPreviewMs = 0;
  if (useRealDecode) {
    try {
      // The pass spaces previews by the source's frame rate, which the first frame's duration
      // gives; a track whose durations are unreadable never got this far.
      const framesPerSecond = 1000 / (frameStarts[1] - frameStarts[0] || 1000 / 24);
      previews = await video.previews(new PreviewOptions(framesPerSecond));
      lines.push(
        `Seek previews: ${previews.total} positions, ${previews.stride} frames apart, filling in the background.`,
      );
      pumpPreviews();
    } catch (error) {
      lines.push(`video.previews() rejected with ${errorCode(error)}: a drag walks the group of pictures instead.`);
    }
    status.textContent = lines.join("\n");
  }

  // The pass has no thread to run on here - that is exactly why the native `PreviewIndex` is
  // native-only - so the page lends it the gaps between its own work: one preview per idle
  // callback, yielding to the event loop in between, and stopping as soon as the pass has
  // visited every position. `requestIdleCallback` is missing from some browsers, where a
  // zero-delay timeout stands in for it.
  function scheduleIdle(callback) {
    if (typeof requestIdleCallback === "function") requestIdleCallback(callback);
    else setTimeout(callback, 0);
  }

  function pumpPreviews() {
    if (!previews || previews.complete || stopped) return;
    scheduleIdle(async () => {
      try {
        await previews.step();
      } catch (error) {
        // A preview that will not decode leaves its position empty and the pass carries on, so
        // the only thing that ends the loop here is the index itself failing.
        previews = null;
        lines.push(`Preview pass stopped with ${errorCode(error) ?? "ERROR"}.`);
        status.textContent = lines.join("\n");
        return;
      }
      pumpPreviews();
    });
  }

  // The seek: what is at this position of the timeline, answered from a picture the pass already
  // decoded. A lookup and an upload, never a decode, whatever the drag jumped over - which is
  // what makes it constant time and what keeps it inside the budget. The walk behind it replaces
  // this with the exact frame shortly.
  function drawPreview(target) {
    if (!previews || !shouldDrawPreview(frameIndex, target)) return;
    const started = performance.now();
    const preview = previews.nearest(BigInt(target));
    if (!preview) return;
    const picture = preview.picture;
    uploadFrame(picture.pixels, picture.width, picture.height);
    const elapsed = performance.now() - started;
    worstPreviewMs = Math.max(worstPreviewMs, elapsed);
    if (seekLatency) {
      seekLatency.textContent =
        `seek ${elapsed.toFixed(2)} ms (worst ${worstPreviewMs.toFixed(2)} of ${budgetMs} ms)`;
    }
  }

  async function frameDuration(index) {
    try {
      return await video.frameDuration(BigInt(index));
    } catch {
      // The previous frame was the last indexed frame, so loop its timing with the video.
      frameIndex = 0;
      return await video.frameDuration(0n);
    }
  }

  async function buildFrameStarts(video) {
    const starts = [0];
    for (let index = 0; ; index++) {
      try {
        starts.push(starts[index] + (await video.frameDuration(BigInt(index))));
      } catch {
        return starts;
      }
    }
  }

  function frameForMediaTime(milliseconds) {
    let low = 0;
    let high = Math.max(0, frameStarts.length - 2);
    while (low < high) {
      const mid = Math.floor((low + high + 1) / 2);
      if (frameStarts[mid] <= milliseconds) low = mid;
      else high = mid - 1;
    }
    return Math.min(low, frameStarts.length - 1);
  }

  function mediaTimeForFrame(index) {
    return frameStarts[Math.min(Math.max(index, 0), lastFrameIndex)] ?? 0;
  }

  function syncTimeline() {
    timeline.value = String(Math.min(frameIndex, lastFrameIndex));
  }

  function currentMediaTimeMs() {
    if (!playing) return pausedMediaTimeMs;
    if (audioState?.startedAt !== undefined) {
      // Audio is scheduled a few ms ahead of the clock, so clamp the pre-roll window to zero.
      const elapsedMs = Math.max((audioState.context.currentTime - audioState.startedAt) * 1000, 0);
      return ((elapsedMs % mediaDurationMs) + mediaDurationMs) % mediaDurationMs;
    }
    return (performance.now() - videoClockStart) % mediaDurationMs;
  }

  async function seekToFrame(index, keepPlaying = playing) {
    frameIndex = Math.min(Math.max(index, 0), lastFrameIndex);
    pausedMediaTimeMs = mediaTimeForFrame(frameIndex);
    if (audioState) stopAudio(audioState);
    if (keepPlaying && audioState) await startAudio(audioState, pausedMediaTimeMs);
    if (keepPlaying && !audioState) videoClockStart = performance.now() - pausedMediaTimeMs;
    await renderFrame(false);
    syncTimeline();
  }

  // Pointer scrubbing can outrun the decoder, so keep only the newest requested position and
  // drop superseded ones instead of queueing every intermediate seek.
  let pendingSeek = null;
  let seeking = false;
  async function requestSeek(index, keepPlaying) {
    pendingSeek = { index, keepPlaying: keepPlaying ?? playing };
    if (seeking) return;
    seeking = true;
    try {
      while (pendingSeek) {
        const next = pendingSeek;
        pendingSeek = null;
        await seekToFrame(next.index, next.keepPlaying);
      }
    } finally {
      seeking = false;
    }
  }

  // Dropping a superseded position is not the same as stopping the decode it started. A group of
  // pictures is 768 frames in the bundled sample - its `stss` names one sync sample - so the seek
  // already under way holds the decoder for seconds while the pointer has long moved on, and the
  // newest position only starts once it finishes (issue #333). Scrubbing therefore aborts the
  // request it is replacing: `video.get` takes an `AbortSignal` and checks it on every turn of
  // its decode loop, so the decoder is freed part-way through rather than after.
  //
  // Which request it may abort is the first half of tracking the pointer (issue #363). A decode
  // still on the way to the newest position is work that position needs: cancelling it throws
  // the decoder's place away and sends the walk back to a random-access point, so only a step
  // the pointer has already moved *behind* is aborted.
  //
  // Walking is the other half, and it always runs forwards, because decoding does. A target
  // behind the frame on screen therefore cannot be walked to: the walk restarts at the
  // random-access point at or before it and comes forwards from there. Asking for such a target
  // directly instead decodes that same span while drawing nothing, and the next pointer sample
  // aborted it before it landed, which is what left a backwards drag frozen.
  //
  // How far each step of the walk goes is measured rather than fixed at one frame. Drawing every
  // frame it passes makes the picture crawl behind a fast drag, because each one costs a decode,
  // an RGBA readback and an upload; `native_gl` measured the same thing from the other side and
  // stopped drawing them at all, which is what left *it* frozen (issues #354 and #355). So a
  // step is whatever fits in `SCRUB_INTERVAL_MS` at the rate the walk is actually decoding at:
  // the picture moves about seven times a second whatever the span, and the frames in between
  // are never asked for, so they cost the browser's decoding and nothing of ours.
  let scrubTarget = null;
  let scrubbing = false;
  let scrubAbort = null;
  // The frame the in-flight decode is for, so a newer target can tell whether it has passed it.
  let scrubStep = null;
  // Where the last drawn frame left the decoder, or null when a cancelled step left it between
  // frames and the next walk has to start over at a random-access point.
  let scrubPosition = null;

  // What a frame is costing this walk, measured from the steps it has already taken. The step
  // it buys is `scrubStrideFrames`, and where the walk goes next is `scrubWalkStart`.
  let scrubMsPerFrame = null;

  async function scrubTo(index) {
    // Without a browser HEVC decoder there is nothing to walk through: the synthetic frames are
    // generated from the index, so the ordinary seek draws them at once.
    if (!useRealDecode) return requestSeek(index, false);
    scrubTarget = Math.min(Math.max(index, 0), lastFrameIndex);
    // Answer the seek before starting the walk, on every pointer sample rather than only on the
    // first: this is the picture the drag is actually following, and it is drawn now instead of
    // after however many frames the exact one is away.
    drawPreview(scrubTarget);
    if (scrubbing) {
      if (scrubStep !== null && scrubStep > scrubTarget) scrubAbort?.abort();
      return;
    }
    scrubbing = true;
    scrubPosition = frameIndex;
    playing = false;
    playButton.textContent = "Play";
    if (audioState) stopAudio(audioState);
    try {
      while (scrubTarget !== null) {
        const target = scrubTarget;
        const next = scrubWalkStart(scrubPosition, target, scrubMsPerFrame, randomAccessPoints);
        scrubAbort = new AbortController();
        scrubStep = next;
        const startedAt = performance.now();
        try {
          const frame = await video.get(BigInt(next), scrubAbort.signal);
          uploadFrame(frame.pixels, frame.width, frame.height);
          // Only a step whose span is known says anything about the rate: a restart at a
          // random-access point decoded an unknown number of frames to get there.
          const span = scrubPosition === null ? 0 : next - scrubPosition;
          if (span > 0) scrubMsPerFrame = (performance.now() - startedAt) / span;
          frameIndex = next;
          scrubPosition = next;
          pausedMediaTimeMs = mediaTimeForFrame(frameIndex);
        } catch (error) {
          // An aborted step is the newest position taking over, not a failure, but it stopped the
          // decoder between frames, so the next walk starts from a random-access point rather
          // than from nowhere. Anything else leaves the picture where it is and ends the scrub.
          if (errorCode(error) !== "CANCELLED") {
            scrubTarget = null;
            break;
          }
          scrubPosition = null;
        } finally {
          scrubStep = null;
        }
        // Arriving clears the target rather than holding it, so a drag that comes back to this
        // frame asks for it afresh - out of the decoder's cache, for nothing.
        if (scrubTarget === target && frameIndex === target) scrubTarget = null;
      }
    } finally {
      scrubbing = false;
      scrubAbort = null;
      scrubStep = null;
      syncTimeline();
    }
  }

  async function seekByMilliseconds(deltaMs) {
    const targetTime = ((currentMediaTimeMs() + deltaMs) % mediaDurationMs + mediaDurationMs) % mediaDurationMs;
    await requestSeek(frameForMediaTime(targetTime));
  }

  async function stepFrame(delta) {
    const from = playing ? frameForMediaTime(currentMediaTimeMs()) : frameIndex;
    playing = false;
    playButton.textContent = "Play";
    await requestSeek(from + delta, false);
  }

  async function renderFrame(advance = true) {
    if (useRealDecode) {
      try {
        const duration = await frameDuration(frameIndex);
        const frame = await video.get(BigInt(frameIndex));
        uploadFrame(frame.pixels, frame.width, frame.height);
        if (advance) frameIndex += 1;
        syncTimeline();
        return duration;
      } catch {
        // Ran past the last indexed frame: loop back to the start.
        frameIndex = 0;
        const duration = await frameDuration(frameIndex);
        const frame = await video.get(0n);
        uploadFrame(frame.pixels, frame.width, frame.height);
        if (advance) frameIndex = 1;
        syncTimeline();
        return duration;
      }
    }
    const duration = await frameDuration(frameIndex);
    const phase = (frameIndex % 60) / 60;
    uploadFrame(syntheticFrame(canvas.width, canvas.height, phase), canvas.width, canvas.height);
    if (advance) frameIndex += 1;
    syncTimeline();
    return duration;
  }

  let framesSinceFpsUpdate = 0;
  let lastFpsUpdate = performance.now();
  let lastDisplayedFrame = -1;

  function tickFps() {
    framesSinceFpsUpdate += 1;
    const now = performance.now();
    const elapsed = now - lastFpsUpdate;
    if (elapsed >= 500) {
      fps.textContent = `${((framesSinceFpsUpdate * 1000) / elapsed).toFixed(1)} fps`;
      framesSinceFpsUpdate = 0;
      lastFpsUpdate = now;
    }
  }

  async function renderLoop() {
    while (!stopped) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      if (playing) {
        frameIndex = frameForMediaTime(currentMediaTimeMs());
        const displayedFrame = frameIndex;
        await renderFrame();
        // Count newly displayed video frames, not render-loop iterations: a
        // display refreshing faster than the source rate re-presents the same
        // frame, which would otherwise read as a frame rate above the clip's.
        if (displayedFrame !== lastDisplayedFrame) {
          lastDisplayedFrame = displayedFrame;
          tickFps();
        }
      }
    }
  }

  playButton.addEventListener("click", async () => {
    playing = !playing;
    if (playing) {
      videoClockStart = performance.now() - pausedMediaTimeMs;
      if (audioState) await startAudio(audioState, pausedMediaTimeMs);
    } else if (audioState) {
      pausedMediaTimeMs = currentMediaTimeMs();
      stopAudio(audioState);
    } else {
      pausedMediaTimeMs = currentMediaTimeMs();
    }
    playButton.textContent = playing ? "Pause" : "Play";
  });
  rewindButton.addEventListener("click", () => seekByMilliseconds(-5000));
  fastForwardButton.addEventListener("click", () => seekByMilliseconds(5000));
  previousFrameButton.addEventListener("click", () => stepFrame(-1));
  nextFrameButton.addEventListener("click", () => stepFrame(1));
  // The range input's `input` event fires on a click and throughout a drag, and only then.
  // Scrubbing on a bare `mousemove` as well meant merely crossing the bar queued a seek per
  // pointer sample, each one re-running the decoder (issue #319).
  timeline.addEventListener("input", () => scrubTo(Number(timeline.value)));

  if (decodedFrame) {
    uploadFrame(decodedFrame.pixels, decodedFrame.width, decodedFrame.height);
    frameIndex = 1;
  } else {
    await renderFrame();
  }
  renderLoop();

  window.addEventListener(
    "pagehide",
    () => {
      stopped = true;
      input.close();
    },
    { once: true },
  );
}

async function prepareAudio(audio) {
  if (!globalThis.AudioDecoder) throw new Error("WebCodecs AudioDecoder is unavailable");
  const config = await audio.aacConfig();
  const packetCount = Number(await audio.packetCount());
  const packets = [];
  for (let index = 0; index < packetCount; index++) {
    packets.push(await audio.packet(BigInt(index)));
  }
  const context = new AudioContext({ sampleRate: config.sampleRate });
  return {
    audio,
    config: {
      codec: config.codec,
      sampleRate: config.sampleRate,
      numberOfChannels: config.channels,
      description: config.audioSpecificConfig,
    },
    context,
    packets,
    buffer: undefined,
    sources: [],
  };
}

/// Decodes every AAC packet once into a single contiguous `AudioBuffer` covering the whole track.
/// Seeking and scrubbing then only reschedule one buffer source instead of re-running the decoder.
async function decodeAudioBuffer(state) {
  if (state.buffer) return state.buffer;
  const outputs = [];
  const decoder = new AudioDecoder({
    output: (data) => outputs.push(data),
    error: (error) => console.error(error),
  });
  decoder.configure(state.config);
  for (const packet of state.packets) {
    const range = packet.range;
    decoder.decode(new EncodedAudioChunk({
      type: "key",
      timestamp: Number(range.start) * 1_000_000 / state.config.sampleRate,
      duration: Number(range.length) * 1_000_000 / state.config.sampleRate,
      data: packet.data,
    }));
  }
  await decoder.flush();
  decoder.close();
  if (outputs.length === 0) throw new Error("AAC decode produced no audio");

  const sampleRate = outputs[0].sampleRate;
  const channels = outputs[0].numberOfChannels;
  const frames = outputs.reduce(
    (end, data) => Math.max(end, sampleAt(data, sampleRate) + data.numberOfFrames),
    0,
  );
  const buffer = state.context.createBuffer(channels, Math.max(frames, 1), sampleRate);
  for (const data of outputs) {
    const start = sampleAt(data, sampleRate);
    for (let channel = 0; channel < channels; channel++) {
      const plane = new Float32Array(data.numberOfFrames);
      data.copyTo(plane, { planeIndex: Math.min(channel, data.numberOfChannels - 1), format: "f32-planar" });
      buffer.copyToChannel(plane, channel, start);
    }
    data.close();
  }
  state.buffer = buffer;
  return buffer;
}

function sampleAt(data, sampleRate) {
  return Math.max(0, Math.round((data.timestamp / 1_000_000) * sampleRate));
}

async function startAudio(state, offsetMs = 0) {
  stopAudio(state);
  const buffer = await decodeAudioBuffer(state);
  await state.context.resume();
  const offsetSeconds = Math.min(Math.max(offsetMs / 1000, 0), Math.max(buffer.duration - 0.001, 0));
  const startAt = state.context.currentTime + 0.05;
  const source = state.context.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  source.connect(state.context.destination);
  source.start(startAt, offsetSeconds);
  state.sources.push(source);
  // Anchor the audio clock to media time zero so `currentMediaTimeMs()` stays absolute across seeks.
  state.startedAt = startAt - offsetSeconds;
}

function stopAudio(state) {
  for (const source of state.sources.splice(0)) {
    try {
      source.stop();
    } catch {}
  }
  state.startedAt = undefined;
}

main().catch((error) => {
  status.textContent = `Error: ${error?.message ?? error}`;
  console.error(error);
});
