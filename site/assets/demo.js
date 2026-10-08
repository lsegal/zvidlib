// The landing page's live demo, running on zvidlib's WebAssembly package (`../pkg`, built by the
// Docs workflow with `wasm-pack build --target web --features web`).
//
// Two panels: a frame-exact scrubber over the bundled Big Buck Bunny AV1 sample, and a recorder
// that encodes an animated canvas to MP4 or WebM and reads a frame back out of the file it wrote.

const SAMPLE_URL = "media/BigBuckBunny.av1.mp4";
const RECORD_FRAMES = 90;
const RECORD_FPS = 30;
const SAMPLE_RATE = 48_000;
const READBACK_FRAME = 45;

// What each encoder is asked about before it is offered, so a codec this browser cannot encode
// is shown as unavailable instead of failing halfway through a recording.
const VIDEO_CODECS = {
  av1: { label: "AV1", probe: "av01.0.04M.08", containers: ["mp4", "webm"] },
  vp9: { label: "VP9", probe: "vp09.00.10.08", containers: ["mp4", "webm"] },
  hevc: { label: "HEVC", probe: "hvc1.1.6.L93.B0", containers: ["mp4"] },
  vp8: { label: "VP8", probe: "vp8", containers: ["webm"] },
};

const $ = (selector) => document.querySelector(selector);
const errorText = (zvid, error) => zvid.errorCode(error) ?? error?.message ?? String(error);

let zvidPromise = null;
function loadZvidlib() {
  zvidPromise ??= import("../pkg/zvidlib.js").then(async (module) => {
    await module.default();
    return module;
  });
  return zvidPromise;
}

// Copies a zvidlib RGBA frame into a canvas, scaling it to fill the canvas.
const scratch = document.createElement("canvas");
function paint(canvas, picture) {
  const { width, height } = picture;
  const image = new ImageData(new Uint8ClampedArray(picture.pixels), width, height);
  const context = canvas.getContext("2d");
  if (width === canvas.width && height === canvas.height) {
    context.putImageData(image, 0, 0);
    return;
  }
  scratch.width = width;
  scratch.height = height;
  scratch.getContext("2d").putImageData(image, 0, 0);
  context.imageSmoothingQuality = "high";
  context.drawImage(scratch, 0, 0, canvas.width, canvas.height);
}

const idle = (callback) =>
  typeof requestIdleCallback === "function" ? requestIdleCallback(callback, { timeout: 200 }) : setTimeout(callback, 16);

// ---------------------------------------------------------------------------------------------
// Frame-exact scrubbing
// ---------------------------------------------------------------------------------------------

async function startScrubber() {
  const canvas = $("#scrub-canvas");
  const badge = $("#scrub-badge");
  const label = $("#scrub-frame");
  const status = $("#scrub-status");
  const timeline = $("#scrub-timeline");
  const play = $("#scrub-play");
  const buttons = [$("#scrub-prev"), play, $("#scrub-next"), $("#scrub-random"), timeline];
  const previewBar = $("#scrub-previews-bar");

  status.textContent = "Loading zvidlib (WebAssembly)…";
  const zvid = await loadZvidlib();
  status.textContent = "Fetching the sample clip…";
  const response = await fetch(SAMPLE_URL);
  if (!response.ok) throw new Error(`${SAMPLE_URL}: HTTP ${response.status}`);
  const input = await zvid.MediaInput.open(await response.blob());
  const video = input.video(0);

  // Frame start times in milliseconds, from the track's own sample durations.
  const starts = [0];
  for (;;) {
    try {
      starts.push(starts.at(-1) + (await video.frameDuration(BigInt(starts.length - 1))));
    } catch {
      break;
    }
  }
  const frameCount = starts.length - 1;
  const last = frameCount - 1;
  timeline.max = String(last);
  const formatTime = (ms) => `${Math.floor(ms / 60000)}:${((ms / 1000) % 60).toFixed(3).padStart(6, "0")}`;

  let shown = -1;
  let wanted = 0;
  let decoding = false;
  let playing = false;
  let playStartedAt = 0;
  let playStartFrame = 0;

  function show(kind, frame, detail) {
    badge.className = `badge ${kind}`;
    badge.textContent = kind === "exact" ? "Exact frame" : "Seek preview";
    label.textContent = `frame ${frame} / ${last} · ${formatTime(starts[frame])}${detail ? ` · ${detail}` : ""}`;
  }

  // Decodes towards the newest wanted frame, one exact request at a time. A drag that outruns
  // the decoder only ever has one request in flight, and the next one goes to wherever the
  // pointer is by then.
  async function pump() {
    if (decoding) return;
    decoding = true;
    try {
      while (shown !== wanted) {
        const target = wanted;
        const began = performance.now();
        const frame = await video.get(BigInt(target));
        const elapsed = performance.now() - began;
        paint(canvas, frame);
        frame.free();
        shown = target;
        show("exact", target, playing ? null : `decoded in ${elapsed.toFixed(1)} ms`);
        if (!playing) timeline.value = String(target);
      }
    } catch (error) {
      status.textContent = `video.get() rejected: ${errorText(zvid, error)}`;
    } finally {
      decoding = false;
    }
  }

  // The seek-preview tier: one downscaled picture every stride frames, filled in idle time on a
  // decoder of its own. A lookup never decodes, so a drag draws something immediately.
  let previews = null;
  try {
    const options = new zvid.PreviewOptions(1000 / (starts[1] - starts[0] || 1000 / 24));
    previews = await video.previews(options);
    options.free();
    const fill = () => {
      if (!previews || previews.complete) {
        previewBar.style.width = "100%";
        return;
      }
      idle(async () => {
        try {
          await previews.step();
          previewBar.style.width = `${(100 * previews.filled) / previews.total}%`;
          fill();
        } catch {
          previews = null;
        }
      });
    };
    fill();
  } catch {
    previews = null;
  }

  function seek(frame) {
    wanted = Math.max(0, Math.min(last, frame));
    if (previews && wanted !== shown) {
      const preview = previews.nearest(BigInt(wanted));
      if (preview) {
        const picture = preview.picture;
        paint(canvas, picture);
        show("preview", Number(preview.frame), `exact frame ${wanted} decoding…`);
        picture.free();
        preview.free();
      }
    }
    pump();
  }

  function tick(now) {
    if (!playing) return;
    const ms = starts[playStartFrame] + (now - playStartedAt);
    let frame = playStartFrame;
    while (frame < last && starts[frame + 1] <= ms) frame++;
    if (frame >= last) {
      playStartFrame = 0;
      playStartedAt = now;
      frame = 0;
    }
    if (frame !== wanted) {
      wanted = frame;
      timeline.value = String(frame);
      pump();
    }
    requestAnimationFrame(tick);
  }

  function setPlaying(next) {
    playing = next;
    play.textContent = playing ? "Pause" : "Play";
    if (playing) {
      playStartFrame = shown < 0 || shown >= last ? 0 : shown;
      playStartedAt = performance.now();
      requestAnimationFrame(tick);
    }
  }

  play.addEventListener("click", () => setPlaying(!playing));
  $("#scrub-prev").addEventListener("click", () => (setPlaying(false), seek(shown - 1)));
  $("#scrub-next").addEventListener("click", () => (setPlaying(false), seek(shown + 1)));
  $("#scrub-random").addEventListener("click", () => (setPlaying(false), seek(Math.floor(Math.random() * frameCount))));
  timeline.addEventListener("input", () => (setPlaying(false), seek(Number(timeline.value))));
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") seek(shown - 1);
    if (event.key === "ArrowRight") seek(shown + 1);
  });

  status.textContent =
    `Opened ${(Number(input.byteLength) / 1048576).toFixed(1)} MiB ${input.container?.toUpperCase()} · ` +
    `${frameCount} frames · AV1 ${canvas.width}x${canvas.height}` +
    (previews ? ` · seek previews every ${previews.stride} frames, filling in the background` : "");
  for (const control of buttons) control.disabled = false;
  seek(0);
}

// ---------------------------------------------------------------------------------------------
// Recording a canvas
// ---------------------------------------------------------------------------------------------

// The animation the recorder encodes. It paints its own frame number, so the frame read back
// out of the finished file can be checked by eye.
function drawScene(context, index) {
  const { width, height } = context.canvas;
  const t = index / RECORD_FPS;
  const background = context.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, `hsl(${250 + 40 * Math.sin(t * 1.3)}, 70%, 16%)`);
  background.addColorStop(1, `hsl(${190 + 40 * Math.cos(t)}, 80%, 22%)`);
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);

  for (let i = 0; i < 14; i++) {
    const angle = t * (0.6 + i * 0.07) + (i * Math.PI * 2) / 14;
    const radius = 70 + 50 * Math.sin(t * 2 + i);
    const x = width / 2 + Math.cos(angle) * radius * 1.7;
    const y = height / 2 + Math.sin(angle) * radius;
    context.beginPath();
    context.arc(x, y, 10 + 6 * Math.sin(t * 3 + i), 0, Math.PI * 2);
    context.fillStyle = `hsla(${(i * 26 + index * 4) % 360}, 90%, 65%, 0.85)`;
    context.fill();
  }

  context.fillStyle = "rgba(0, 0, 0, 0.45)";
  context.fillRect(0, height - 64, width, 64);
  context.fillStyle = "#fff";
  context.font = "700 34px ui-monospace, Menlo, Consolas, monospace";
  context.textBaseline = "middle";
  context.fillText(`frame ${String(index).padStart(3, "0")}`, 22, height - 32);
  context.font = "600 22px system-ui, sans-serif";
  context.textAlign = "right";
  context.fillText("zvidlib", width - 22, height - 32);
  context.textAlign = "left";
}

// A C major chord with a soft attack each second, interleaved stereo.
function chord(startSample, length) {
  const samples = new Float32Array(length * 2);
  for (let i = 0; i < length; i++) {
    const n = startSample + i;
    const time = n / SAMPLE_RATE;
    const envelope = Math.min(1, (time % 1) * 8) * Math.exp(-(time % 1) * 2.5);
    const value =
      0.12 * envelope *
      (Math.sin(2 * Math.PI * 261.63 * time) + Math.sin(2 * Math.PI * 329.63 * time) + Math.sin(2 * Math.PI * 392 * time));
    samples[i * 2] = value;
    samples[i * 2 + 1] = value;
  }
  return samples;
}

async function encodableCodecs() {
  const available = new Set();
  if (!globalThis.VideoEncoder) return available;
  await Promise.all(
    Object.entries(VIDEO_CODECS).map(async ([name, codec]) => {
      try {
        const support = await VideoEncoder.isConfigSupported({ codec: codec.probe, width: 640, height: 360, framerate: RECORD_FPS });
        if (support.supported) available.add(name);
      } catch {
        // A codec string the browser cannot parse is a no.
      }
    }),
  );
  return available;
}

async function startRecorder() {
  const canvas = $("#record-canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const containerSelect = $("#record-container");
  const codecSelect = $("#record-codec");
  const audioToggle = $("#record-audio");
  const start = $("#record-start");
  const status = $("#record-status");
  const bar = $("#record-bar");
  const player = $("#record-video");
  const download = $("#record-download");
  const readback = $("#readback-canvas");
  const readbackText = $("#readback-text");

  // Animate the canvas while idle, so the panel shows what will be recorded.
  let preview = 0;
  let recording = false;
  const animate = () => {
    if (!recording) drawScene(context, preview++ % RECORD_FRAMES);
    setTimeout(() => requestAnimationFrame(animate), 1000 / RECORD_FPS);
  };
  animate();

  status.textContent = "Loading zvidlib (WebAssembly)…";
  const zvid = await loadZvidlib();
  const available = await encodableCodecs();

  function fillCodecs() {
    const container = containerSelect.value;
    codecSelect.replaceChildren();
    for (const [name, codec] of Object.entries(VIDEO_CODECS)) {
      if (!codec.containers.includes(container)) continue;
      const usable = available.has(name) && zvid.videoEncodeSupport("prefer", name);
      const option = new Option(usable ? codec.label : `${codec.label} (not in this browser)`, name);
      option.disabled = !usable;
      codecSelect.append(option);
    }
    const first = [...codecSelect.options].find((option) => !option.disabled);
    if (first) first.selected = true;
    start.disabled = !first;
    status.textContent = first
      ? "Ready. Pick a container and codec, then encode."
      : "This browser has no WebCodecs video encoder for this container.";
  }
  containerSelect.addEventListener("change", fillCodecs);
  fillCodecs();

  start.addEventListener("click", async () => {
    recording = true;
    start.disabled = true;
    download.hidden = true;
    const container = containerSelect.value;
    let output = null;
    try {
      const options = new zvid.CreateOptions(container);
      options.videoCodec = codecSelect.value;
      options.setTimeline(RECORD_FPS, 1, SAMPLE_RATE);
      let audioCodec = null;
      if (audioToggle.checked) {
        audioCodec = container === "mp4" && zvid.audioEncodeSupport("prefer", "aac") ? "aac" : "opus";
        options.audioCodec = audioCodec;
      }
      output = await zvid.MediaOutput.create(options);
      const video = output.video(0);
      const audio = audioCodec ? output.audio(0) : null;
      const samplesPerFrame = SAMPLE_RATE / RECORD_FPS;
      const began = performance.now();

      for (let index = 0; index < RECORD_FRAMES; index++) {
        drawScene(context, index);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        await video.put(BigInt(index), zvid.VideoFrame.rgba(canvas.width, canvas.height, pixels));
        if (audio) {
          const first = index * samplesPerFrame;
          const range = new zvid.SampleRange(BigInt(first), BigInt(first + samplesPerFrame));
          await audio.put(BigInt(index), new zvid.AudioBuffer(range, SAMPLE_RATE, 2, chord(first, samplesPerFrame)));
        }
        bar.style.width = `${(100 * (index + 1)) / RECORD_FRAMES}%`;
        status.textContent = `put(${index}) · encoding frame ${index + 1} of ${RECORD_FRAMES}…`;
      }

      status.textContent = "finish() · draining encoders and writing the index…";
      const blob = await output.finish();
      output = null;
      const seconds = (performance.now() - began) / 1000;
      const codecLabel = VIDEO_CODECS[codecSelect.value].label;
      const audioLabel = audioCodec ? ` + ${audioCodec === "aac" ? "AAC" : "Opus"}` : "";
      status.textContent =
        `Wrote ${(blob.size / 1024).toFixed(1)} KiB ${container.toUpperCase()} (${codecLabel}${audioLabel}) ` +
        `in ${seconds.toFixed(1)} s: ${RECORD_FRAMES} frames at ${RECORD_FPS} fps.`;

      if (player.src) URL.revokeObjectURL(player.src);
      const url = URL.createObjectURL(blob);
      player.src = url;
      player.muted = !audioCodec;
      player.play().catch(() => {});
      download.href = url;
      download.download = `zvidlib-demo.${container}`;
      download.textContent = `Download ${container.toUpperCase()} (${(blob.size / 1024).toFixed(1)} KiB)`;
      download.hidden = false;

      // Read a frame back out of the file that was just written, through the same exact path the
      // scrubber uses.
      const input = await zvid.MediaInput.open(blob);
      try {
        const frame = await input.video(0).get(BigInt(READBACK_FRAME));
        paint(readback, frame);
        frame.free();
        readbackText.textContent =
          `zvidlib reopened the ${input.container?.toUpperCase()} it just wrote and read frame ${READBACK_FRAME} back. ` +
          `The counter should say ${String(READBACK_FRAME).padStart(3, "0")}.`;
      } catch (error) {
        readbackText.textContent = `Reading frame ${READBACK_FRAME} back failed: ${errorText(zvid, error)}.`;
      } finally {
        input.close();
      }
    } catch (error) {
      status.textContent = `Encoding failed: ${errorText(zvid, error)}`;
      output?.close();
    } finally {
      recording = false;
      start.disabled = false;
    }
  });
}

// Load the demo when it scrolls into view, or when a link jumps to it, rather than making every
// visitor download the WebAssembly module up front.
let scrubberStarted = false;
let recorderStarted = false;
function startPanel(name) {
  if (name === "record" && !recorderStarted) {
    recorderStarted = true;
    startRecorder().catch((error) => ($("#record-status").textContent = `Demo failed to start: ${error?.message ?? error}`));
  }
  if (name === "scrub" && !scrubberStarted) {
    scrubberStarted = true;
    startScrubber().catch((error) => {
      $("#scrub-status").textContent = `Demo failed to start: ${error?.message ?? error}`;
      $("#scrub-badge").textContent = "Unavailable";
    });
  }
}

const demo = $("#demo");
new IntersectionObserver(
  (entries, observer) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    startPanel($("#tab-record").getAttribute("aria-selected") === "true" ? "record" : "scrub");
  },
  { rootMargin: "300px" },
).observe(demo);
$("#tab-scrub").addEventListener("tabselected", () => startPanel("scrub"));
$("#tab-record").addEventListener("tabselected", () => startPanel("record"));
