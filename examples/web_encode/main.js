import init, { AudioBuffer, CreateOptions, MediaOutput, SampleRange, VideoFrame, audioEncodeSupport, videoEncodeSupport } from "./pkg/zvidlib.js";

const encodeButton = document.querySelector("#encode");
const download = document.querySelector("#download");
const preview = document.querySelector("#preview");
const status = document.querySelector("#status");
const width = 320, height = 180, frameRate = 30, sampleRate = 48_000;

function pixelsForFrame(index) {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const offset = (y * width + x) * 4;
    pixels[offset] = (x + index * 7) & 255;
    pixels[offset + 1] = (y * 2 + index * 3) & 255;
    pixels[offset + 2] = (x + y + index * 11) & 255;
    pixels[offset + 3] = 255;
  }
  return pixels;
}

function tone(start, frames) {
  const samples = new Float32Array(frames * 2);
  for (let frame = 0; frame < frames; frame += 1) {
    const value = Math.sin(2 * Math.PI * 440 * (start + frame) / sampleRate) * 0.15;
    samples[frame * 2] = value; samples[frame * 2 + 1] = value;
  }
  return samples;
}

await init();
const videoAvailable = videoEncodeSupport();
const audioAvailable = audioEncodeSupport();
status.textContent = videoAvailable ? `Video encoding is available. AAC-LC audio: ${audioAvailable ? "available" : "unavailable"}.` : "This browser does not expose the WebCodecs AV1 encoder required by this example.";
encodeButton.disabled = !videoAvailable;

encodeButton.addEventListener("click", async () => {
  encodeButton.disabled = true; download.hidden = true;
  try {
    const options = new CreateOptions("mp4"); options.setTimeline(frameRate, 1, sampleRate);
    const output = await MediaOutput.create(options);
    const video = output.video(0), audio = audioAvailable ? output.audio(0) : null;
    const samplesPerFrame = sampleRate / frameRate;
    for (let index = 0; index < frameRate; index += 1) {
      await video.put(BigInt(index), VideoFrame.rgba(width, height, pixelsForFrame(index)));
      if (audio) {
        const start = index * samplesPerFrame;
        const range = new SampleRange(BigInt(start), BigInt(start + samplesPerFrame));
        await audio.put(BigInt(index), new AudioBuffer(range, sampleRate, 2, tone(start, samplesPerFrame)));
      }
      status.textContent = `Encoding frame ${index + 1} of ${frameRate}…`;
    }
    const blob = await output.finish(), url = URL.createObjectURL(blob);
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = url; download.href = url; download.hidden = false;
    status.textContent = `Created ${blob.size.toLocaleString()} bytes of MP4${audio ? " with synchronized AAC" : " (video only)"}.`;
  } catch (error) {
    status.textContent = `Encoding failed: ${error.message || error}`;
  } finally { encodeButton.disabled = false; }
});
