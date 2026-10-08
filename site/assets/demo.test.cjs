const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");

const source = readFileSync(`${__dirname}/demo.js`, "utf8");
const settle = () => new Promise(setImmediate);

async function scrubber({ audio = true, delayVideo = false } = {}) {
  const elements = new Map();
  const windowEvents = new Map();
  const frames = [];
  const audioStarts = [];
  const sources = [];
  const pendingFrames = [];
  const gain = { value: 1, setTargetAtTime(value) { this.value = value; } };
  function element(selector) {
    if (!elements.has(selector)) {
      const listeners = new Map();
      elements.set(selector, {
        width: 1, height: 1, value: "0", style: {},
        addEventListener(name, callback) { listeners.set(name, callback); },
        emit(name, event = {}) { listeners.get(name)?.(event); },
        setAttribute() {},
        getContext() { return { putImageData() {} }; },
      });
    }
    return elements.get(selector);
  }
  const picture = () => ({ width: 1, height: 1, pixels: new Uint8Array(4), free() {} });
  const video = {
    async frameDuration(index) {
      if (index >= 100n) throw new Error("end");
      return 100;
    },
    async get(index) {
      frames.push(Number(index));
      if (delayVideo) await new Promise(resolve => pendingFrames.push(resolve));
      return picture();
    },
    async previews() { throw new Error("no previews"); },
  };
  class AudioContext {
    currentTime = 0;
    resume() {}
    createGain() { return { gain, connect() {} }; }
    createBuffer(channels, length, rate) {
      return { duration: length / rate, getChannelData: () => new Float32Array(length) };
    }
    createBufferSource() {
      const source = { connect() {}, start() {}, stop() { this.stopped = true; } };
      sources.push(source);
      return source;
    }
  }
  const input = {
    byteLength: 100, container: "mp4", video: () => video,
    audio() {
      if (!audio) throw new Error("no audio");
      return {
        async decoderConfig() { return { codec: "mp4a", sampleRate: 100, numberOfChannels: 1 }; },
        async sampleCount() { return 1000n; },
        async getRange(first, end) {
          audioStarts.push(Number(first));
          return { samples: new Float32Array(Number(end - first)), channels: 1, free() {} };
        },
      };
    },
  };
  const sandbox = {
    document: { querySelector: element, createElement: () => element("scratch") },
    addEventListener: (name, callback) => windowEvents.set(name, callback),
    IntersectionObserver: class { observe() {} },
    ImageData: class {}, AudioContext,
    fetch: async () => ({ ok: true, blob: async () => ({}) }),
    performance: { now: () => 0 },
    requestAnimationFrame: () => 1, cancelAnimationFrame() {}, setTimeout() {},
    zvidMock: { MediaInput: { open: async () => input }, errorCode: () => null },
  };
  await runInNewContext(`${source}\nzvidPromise = Promise.resolve(zvidMock); startScrubber();`, sandbox);
  await settle();
  const timeline = element("#scrub-timeline");
  return {
    play: element("#scrub-play"), mute: element("#scrub-mute"), timeline,
    frames, audioStarts, sources, gain,
    press() { timeline.emit("pointerdown", { isPrimary: true, button: 0 }); },
    release(name = "pointerup") { windowEvents.get(name)(); },
    async seek(frame) { timeline.value = String(frame); timeline.emit("input"); await settle(); },
    async decode() { pendingFrames.shift()(); await settle(); },
  };
}

for (const ending of ["pointerup", "pointercancel", "blur"]) {
  test(`playing scrub suspends audio and resumes from the requested frame on ${ending}`, async () => {
    const demo = await scrubber();
    demo.play.emit("click");
    await settle();
    demo.press();
    assert.equal(demo.play.textContent, "Play");
    assert.ok(demo.sources.every(source => source.stopped));
    await demo.seek(42);
    assert.equal(demo.frames.at(-1), 42);
    demo.timeline.value = "10";
    const before = demo.audioStarts.length;
    demo.release(ending);
    await settle();
    assert.equal(demo.play.textContent, "Pause");
    assert.equal(demo.audioStarts[before], 420);
    const after = demo.audioStarts.length;
    demo.release();
    await settle();
    assert.equal(demo.audioStarts.length, after);
  });
}

test("paused pointer and keyboard seeks stay paused", async () => {
  const demo = await scrubber();
  demo.press();
  await demo.seek(25);
  demo.release();
  await demo.seek(26);
  assert.notEqual(demo.play.textContent, "Pause");
  assert.equal(demo.frames.at(-1), 26);
  assert.equal(demo.audioStarts.length, 0);
});

test("keyboard seeks restart playing audio immediately and preserve mute", async () => {
  const demo = await scrubber();
  demo.play.emit("click");
  await settle();
  demo.mute.emit("click");
  const before = demo.audioStarts.length;
  await demo.seek(31);
  assert.equal(demo.audioStarts[before], 310);
  assert.equal(demo.play.textContent, "Pause");
  assert.equal(demo.gain.value, 0);
  demo.press();
  await demo.seek(50);
  demo.release();
  assert.equal(demo.gain.value, 0);
});

test("release during a pending decode resumes from the newest requested frame", async () => {
  const demo = await scrubber({ delayVideo: true });
  await demo.decode();
  demo.play.emit("click");
  demo.press();
  await demo.seek(20);
  await demo.seek(40);
  await demo.decode();
  const before = demo.audioStarts.length;
  demo.release();
  await settle();
  assert.equal(demo.audioStarts[before], 400);
  await demo.decode();
  assert.equal(demo.frames.at(-1), 40);
});

test("video-only playback resumes after a scrub", async () => {
  const demo = await scrubber({ audio: false });
  demo.play.emit("click");
  demo.press();
  await demo.seek(15);
  demo.release();
  assert.equal(demo.play.textContent, "Pause");
  assert.equal(demo.audioStarts.length, 0);
});
