// Node's built-in runner, so this needs no dependency the example does not already have:
// `node --test examples/web_canvas/`. These are the web copy of the rules
// `examples/native_gl/scrub.rs` asserts in `a_stride_is_what_fits_in_one_publishing_interval`,
// `a_walk_continues_forwards_and_restarts_backwards` and
// `a_target_snaps_back_to_the_random_access_point_that_decodes_it` - the two examples make the
// same decisions and only one of them was checked by anything but reading it (issue #380).
import test from "node:test";
import assert from "node:assert/strict";

import {
  SCRUB_INTERVAL_MS,
  SCRUB_MAXIMUM_STEP,
  randomAccessPointAtOrBefore,
  scrubStrideFrames,
  scrubWalkStart,
  shouldDrawPreview,
} from "./scrub.js";

// Every fourth frame, as `samples(12, 4)` builds for the native tests.
const KEYFRAMES = [0, 4, 8];

test("a target snaps back to the random-access point that decodes it", () => {
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 0), 0);
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 3), 0);
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 4), 4);
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 7), 4);
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 11), 8);
  // Past the last indexed frame the caller's own target is the best answer available.
  assert.equal(randomAccessPointAtOrBefore(KEYFRAMES, 40), 8);
});

test("a track without a leading random-access point returns the target itself", () => {
  // `main.js` falls back to `[0]` when the track indexes none, so this is the shape a build
  // that cannot read the sync samples produces rather than an empty list.
  assert.equal(randomAccessPointAtOrBefore([6], 2), 2);
  assert.equal(randomAccessPointAtOrBefore([], 2), 2);
});

test("a stride is what fits in one publishing interval", () => {
  // Nothing measured yet: draw the first picture immediately and measure from it.
  assert.equal(scrubStrideFrames(null), 1);
  // 30 ms a frame fits five of them in the 150 ms interval.
  assert.equal(scrubStrideFrames(30), 5);
  assert.equal(scrubStrideFrames(SCRUB_INTERVAL_MS), 1);
  // A frame slower than the whole interval still moves, one frame at a time.
  assert.equal(scrubStrideFrames(400), 1);
  // And an unmeasurably fast one still draws rather than jumping a whole track blind.
  assert.equal(scrubStrideFrames(0), SCRUB_MAXIMUM_STEP);
  assert.equal(scrubStrideFrames(1e-9), SCRUB_MAXIMUM_STEP);
  // A stride is a whole number of frames, never a fraction of one.
  assert.equal(scrubStrideFrames(7), 21);
});

test("a walk continues forwards and restarts backwards", () => {
  const rate = 30;
  // Ahead of the reader: continue from where it is, one stride at a time.
  assert.equal(scrubWalkStart(2, 11, rate, KEYFRAMES), 7);
  // Never past the target itself.
  assert.equal(scrubWalkStart(2, 5, rate, KEYFRAMES), 5);
  // Already there.
  assert.equal(scrubWalkStart(5, 5, rate, KEYFRAMES), 5);
  // Behind the reader, or a position a cancelled decode left unknown: restart at the
  // random-access point the target decodes from, which is the first frame it can draw.
  assert.equal(scrubWalkStart(11, 6, rate, KEYFRAMES), 4);
  assert.equal(scrubWalkStart(null, 6, rate, KEYFRAMES), 4);
});

test("an unmeasured walk steps one frame and a fast one is still bounded", () => {
  // The first step of any walk, before a span has been timed.
  assert.equal(scrubWalkStart(2, 11, null, KEYFRAMES), 3);
  // The step ceiling applies to the walk, not only to the stride.
  assert.equal(scrubWalkStart(0, 10_000, 0, KEYFRAMES), SCRUB_MAXIMUM_STEP);
});

test("a preview is drawn while the walk is elsewhere and never over the exact frame", () => {
  // The pointer has moved somewhere the walk has not reached: the preview is the only picture
  // of that position there is, and drawing it is what answers the seek inside the budget.
  assert.equal(shouldDrawPreview(0, 500), true);
  assert.equal(shouldDrawPreview(700, 500), true);
  // Nothing drawn yet is still somewhere else.
  assert.equal(shouldDrawPreview(null, 0), true);
  // The walk has landed on the frame the pointer is on, and a downscaled stand-in for it would
  // only blur the exact picture already on the canvas.
  assert.equal(shouldDrawPreview(500, 500), false);
});

test("a single-group-of-pictures track restarts a backwards walk at frame zero", () => {
  // The bundled sample: `stss` names one sync sample, so every backwards drag comes forwards
  // from frame 0 - which is what makes drawing on the way there matter at all (issue #363).
  assert.equal(scrubWalkStart(700, 500, 30, [0]), 0);
  assert.equal(scrubWalkStart(0, 500, 30, [0]), 5);
});
