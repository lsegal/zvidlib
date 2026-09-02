// The pure decisions a scrub walk makes, kept apart from `main.js` so they can be asserted
// without a browser, a wasm build or a decoder - the same rules `examples/native_gl/scrub.rs`
// holds in its unit tests. `main.js` owns the state these are computed from; nothing here reads
// or writes any of it.

// How often the walk draws, and the ceiling on how far one step may jump so that an
// unmeasurably fast decoder still draws pictures rather than skipping the span in silence.
export const SCRUB_INTERVAL_MS = 150;
export const SCRUB_MAXIMUM_STEP = 512;

// The newest random-access frame at or before `index`, or `index` itself when the track
// indexes none before it - the same rule `native_gl`'s `KeyframeIndex` follows.
export function randomAccessPointAtOrBefore(randomAccessPoints, index) {
  let low = 0;
  let high = randomAccessPoints.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (randomAccessPoints[mid] <= index) low = mid + 1;
    else high = mid;
  }
  return low === 0 ? index : randomAccessPoints[low - 1];
}

// How far one step of the walk goes, from what a frame is costing this walk so far.
// Nothing measured yet publishes the first picture immediately and measures from it; a frame
// slower than the whole interval still moves, one frame at a time.
export function scrubStrideFrames(msPerFrame) {
  if (msPerFrame === null) return 1;
  if (msPerFrame <= 0) return SCRUB_MAXIMUM_STEP;
  const frames = Math.floor(SCRUB_INTERVAL_MS / msPerFrame);
  return Math.min(Math.max(frames, 1), SCRUB_MAXIMUM_STEP);
}

// The frame the next step of the walk asks for. Decoding only runs forwards, so a target ahead
// of the reader is continued towards a stride at a time and never overshot, while a target
// behind it - or a position a cancelled decode left unknown - restarts at the random-access
// point the target decodes from.
export function scrubWalkStart(position, target, msPerFrame, randomAccessPoints) {
  if (position === null) return randomAccessPointAtOrBefore(randomAccessPoints, target);
  if (position < target) return Math.min(target, position + scrubStrideFrames(msPerFrame));
  if (position === target) return target;
  return randomAccessPointAtOrBefore(randomAccessPoints, target);
}
