// See examples/README.md for setup: build the wasm package into ./pkg before serving this
// directory over HTTP(S). BigBuckBunny.mp4 is already symlinked in from examples/media/.
import init, { MediaInput, errorCode } from "./pkg/zvidlib.js";

const canvas = document.querySelector("#video");
const playButton = document.querySelector("#play");
const status = document.querySelector("#status");
const fps = document.querySelector("#fps");

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
  status.textContent = lines.join("\n");

  let playing = false;
  let frameIndex = 0;
  let stopped = false;
  let videoClockStart = 0;

  function uploadFrame(pixels, width, height) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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

  async function renderFrame() {
    if (useRealDecode) {
      try {
        const duration = await frameDuration(frameIndex);
        const frame = await video.get(BigInt(frameIndex));
        uploadFrame(frame.pixels, frame.width, frame.height);
        frameIndex += 1;
        return duration;
      } catch {
        // Ran past the last indexed frame: loop back to the start.
        frameIndex = 0;
        const duration = await frameDuration(frameIndex);
        const frame = await video.get(0n);
        uploadFrame(frame.pixels, frame.width, frame.height);
        frameIndex = 1;
        return duration;
      }
    }
    const duration = await frameDuration(frameIndex);
    const phase = (frameIndex % 60) / 60;
    uploadFrame(syntheticFrame(canvas.width, canvas.height, phase), canvas.width, canvas.height);
    frameIndex += 1;
    return duration;
  }

  let framesSinceFpsUpdate = 0;
  let lastFpsUpdate = performance.now();

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
        if (audioState?.startedAt !== undefined) {
          const elapsedMs = (audioState.context.currentTime - audioState.startedAt) * 1000;
          const durationMs = frameStarts[frameStarts.length - 1] || 1;
          frameIndex = frameForMediaTime(((elapsedMs % durationMs) + durationMs) % durationMs);
        } else {
          frameIndex = frameForMediaTime((performance.now() - videoClockStart) % (frameStarts[frameStarts.length - 1] || 1));
        }
        await renderFrame();
        tickFps();
      }
    }
  }

  playButton.addEventListener("click", async () => {
    playing = !playing;
    if (playing) {
      videoClockStart = performance.now();
      if (audioState) await startAudio(audioState);
    } else if (audioState) {
      stopAudio(audioState);
    }
    playButton.textContent = playing ? "Pause" : "Play";
  });

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
    sources: [],
  };
}

async function startAudio(state) {
  stopAudio(state);
  await state.context.resume();
  const startAt = state.context.currentTime + 0.08;
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
  state.startedAt = startAt;
  for (const data of outputs) {
    const buffer = state.context.createBuffer(data.numberOfChannels, data.numberOfFrames, data.sampleRate);
    for (let channel = 0; channel < data.numberOfChannels; channel++) {
      const plane = new Float32Array(data.numberOfFrames);
      data.copyTo(plane, { planeIndex: channel, format: "f32-planar" });
      buffer.copyToChannel(plane, channel);
    }
    const source = state.context.createBufferSource();
    source.buffer = buffer;
    source.connect(state.context.destination);
    source.start(startAt + data.timestamp / 1_000_000);
    state.sources.push(source);
    data.close();
  }
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
