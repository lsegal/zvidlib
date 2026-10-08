import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../../site/assets/demo.js", import.meta.url), "utf8");
const settle = () => new Promise((resolve) => setImmediate(resolve));

async function scrubber({ previews = true } = {}) {
  const elements = new Map();
  const painted = [];
  const pending = [];
  const freed = [];
  const ticks = [];
  const element = (selector) => {
    if (!elements.has(selector)) {
      elements.set(selector, {
        value: "0", style: {}, listeners: {}, width: 1, height: 1,
        addEventListener(name, handler) { this.listeners[name] = handler; },
        setAttribute() {},
        getContext() { return { putImageData(image) { painted.push(image.data[0]); } }; },
      });
    }
    return elements.get(selector);
  };
  const picture = (frame) => ({
    width: 1, height: 1, pixels: [frame, 0, 0, 255],
    free() { freed.push(frame); },
  });
  const video = {
    async frameDuration(frame) { if (frame >= 100n) throw new Error("end"); return 40; },
    get(frame) { return new Promise((resolve) => pending.push({ frame: Number(frame), resolve })); },
    async previews() {
      if (!previews) throw new Error("no previews");
      return {
        complete: true, stride: 1,
        nearest(frame) { return { frame, picture: picture(Number(frame)), free() {} }; },
      };
    },
  };
  const context = vm.createContext({
    document: { querySelector: element, createElement: () => element("scratch") },
    IntersectionObserver: class { observe() {} },
    ImageData: class { constructor(data) { this.data = data; } },
    performance: { now: () => 0 },
    requestAnimationFrame: (callback) => ticks.push(callback),
    fetch: async () => ({ ok: true, blob: async () => ({}) }),
    setTimeout,
  });
  vm.runInContext(source, context);
  context.library = {
    MediaInput: { open: async () => ({ video: () => video, byteLength: 1, container: "mp4" }) },
    PreviewOptions: class { free() {} },
    errorCode: () => null,
  };
  await vm.runInContext("zvidPromise = Promise.resolve(library); startScrubber()", context);
  const complete = async (expected) => {
    const request = pending.shift();
    assert.equal(request?.frame, expected);
    request.resolve(picture(expected));
    await settle();
  };
  const seek = (frame) => {
    element("#scrub-timeline").value = String(frame);
    element("#scrub-timeline").listeners.input();
  };
  await complete(0);
  painted.length = 0;
  return { element, painted, pending, freed, ticks, seek, complete };
}

test("fast scrubbing discards obsolete exact frames and preserves the latest preview and slider", async () => {
  const demo = await scrubber();
  demo.seek(20);
  demo.seek(80);
  await demo.complete(20);
  assert.deepEqual(demo.painted, [20, 80]);
  assert.equal(demo.freed.filter((frame) => frame === 20).length, 2);
  assert.equal(demo.element("#scrub-timeline").value, "80");
  assert.equal(demo.element("#scrub-badge").textContent, "Seek preview");
  await demo.complete(80);
  assert.deepEqual(demo.painted, [20, 80, 80]);
  assert.equal(demo.element("#scrub-badge").textContent, "Exact frame");
  assert.match(demo.element("#scrub-frame").textContent, /^frame 80 /);
});

test("returning to the previous exact position replaces the intervening preview", async () => {
  const demo = await scrubber();
  demo.seek(20);
  demo.seek(0);
  await demo.complete(20);
  await demo.complete(0);
  assert.deepEqual(demo.painted, [20, 0, 0]);
  assert.equal(demo.element("#scrub-badge").textContent, "Exact frame");
  assert.equal(demo.element("#scrub-timeline").value, "0");
});

test("scrubbing without previews drops stale frames and settles on the final target", async () => {
  const demo = await scrubber({ previews: false });
  demo.seek(70);
  demo.seek(10);
  await demo.complete(70);
  assert.deepEqual(demo.painted, []);
  await demo.complete(10);
  assert.deepEqual(demo.painted, [10]);
});

test("slow scrubbing and step buttons keep the slider and exact frame in sync", async () => {
  const demo = await scrubber();
  demo.seek(10);
  await demo.complete(10);
  for (const [button, frame] of [["next", 11], ["prev", 10]]) {
    demo.element(`#scrub-${button}`).listeners.click();
    assert.equal(demo.element("#scrub-timeline").value, String(frame));
    await demo.complete(frame);
  }
  demo.element("#scrub-random").listeners.click();
  const target = Number(demo.element("#scrub-timeline").value);
  if (demo.pending.length) await demo.complete(target);
  assert.match(demo.element("#scrub-frame").textContent, new RegExp(`^frame ${target} `));
});

test("playback still presents decoded frames when the clock advances during decoding", async () => {
  const demo = await scrubber();
  demo.element("#scrub-play").listeners.click();
  demo.ticks.shift()(40);
  demo.ticks.shift()(120);
  await demo.complete(1);
  assert.deepEqual(demo.painted, [1]);
  assert.equal(demo.element("#scrub-timeline").value, "3");
  await demo.complete(3);
  assert.deepEqual(demo.painted, [1, 3]);
});

test("a seek supersedes an in-flight playback frame", async () => {
  const demo = await scrubber();
  demo.element("#scrub-play").listeners.click();
  demo.ticks.shift()(40);
  demo.seek(80);
  await demo.complete(1);
  assert.deepEqual(demo.painted, [80]);
  assert.equal(demo.element("#scrub-timeline").value, "80");
  await demo.complete(80);
  assert.deepEqual(demo.painted, [80, 80]);
});
