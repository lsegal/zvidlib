// See examples/README.md for setup: build the wasm package into ./pkg before serving this
// directory over HTTP(S). BigBuckBunny.mp4 is already symlinked in from examples/media/.
import init, { MediaInput, errorCode } from "./pkg/zvidlib.js";

const canvas = document.querySelector("#video");
const playButton = document.querySelector("#play");
const status = document.querySelector("#status");
const fps = document.querySelector("#fps");
const SAMPLE_FRAME_RATE = 24;

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

  const mediaBlob = await response.blob();
  const mediaUrl = URL.createObjectURL(mediaBlob);
  const audio = new Audio(mediaUrl);
  audio.loop = true;
  audio.preload = "auto";
  const input = await MediaInput.open(mediaBlob);
  const video = input.video(0);
  const lines = [
    `Opened ${input.byteLength} bytes.`,
    `Video stream 0 direction: ${video.direction}`,
    "The MP4 audio track will play through the browser's native audio output.",
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
  status.textContent = lines.join("\n");

  let playing = false;
  let frameIndex = 0;
  let renderedFrameIndex = -1;
  let stopped = false;

  function uploadFrame(pixels, width, height) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  async function renderFrame() {
    const targetFrame = playing
      ? Math.floor(audio.currentTime * SAMPLE_FRAME_RATE)
      : frameIndex;
    if (targetFrame === renderedFrameIndex) return false;
    if (useRealDecode) {
      try {
        const frame = await video.get(BigInt(targetFrame));
        uploadFrame(frame.pixels, frame.width, frame.height);
        renderedFrameIndex = targetFrame;
        frameIndex = targetFrame + 1;
        return true;
      } catch {
        // Ran past the last indexed frame: loop back to the start.
        frameIndex = 0;
        const frame = await video.get(0n);
        uploadFrame(frame.pixels, frame.width, frame.height);
        frameIndex = 1;
        renderedFrameIndex = 0;
        return true;
      }
    }
    const phase = (targetFrame % 60) / 60;
    uploadFrame(syntheticFrame(canvas.width, canvas.height, phase), canvas.width, canvas.height);
    renderedFrameIndex = targetFrame;
    frameIndex = targetFrame + 1;
    return true;
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
      if (playing) {
        if (await renderFrame()) tickFps();
      }
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  playButton.addEventListener("click", async () => {
    if (playing) {
      audio.pause();
      playing = false;
    } else {
      await audio.play();
      playing = true;
    }
    playButton.textContent = playing ? "Pause" : "Play";
  });

  if (decodedFrame) {
    uploadFrame(decodedFrame.pixels, decodedFrame.width, decodedFrame.height);
    renderedFrameIndex = 0;
    frameIndex = 1;
  } else {
    await renderFrame();
  }
  renderLoop();

  window.addEventListener(
    "pagehide",
    () => {
      stopped = true;
      audio.pause();
      URL.revokeObjectURL(mediaUrl);
      input.close();
    },
    { once: true },
  );
}

main().catch((error) => {
  status.textContent = `Error: ${error?.message ?? error}`;
  console.error(error);
});
