// Node's built-in runner, like `scrub.test.js`, so this needs no dependency the example does not
// already have: `node --test 'examples/web_canvas/*.test.js'`. What it covers is the choice
// issue #441 added - which bundled sample the page opens, given what the browser said it can
// decode - kept pure in `samples.js` so it can be asserted without a browser, a wasm build or a
// decoder.
//
// The declarations themselves are checked elsewhere and deliberately not here: asserting a codec
// string against a copy of itself proves nothing, so the file each entry describes is demuxed and
// compared against it by `tests/web_canvas_samples_match_their_files.rs`.
import test from "node:test";
import assert from "node:assert/strict";

import { SAMPLES, decoderConfig, supportedSamples } from "./samples.js";

const HEVC = SAMPLES.find((sample) => sample.codec.startsWith("hev1."));
const AV1 = SAMPLES.find((sample) => sample.codec.startsWith("av01."));

test("the page carries a sample for a browser that decodes neither codec of the other", () => {
  assert.ok(HEVC, "an HEVC sample is declared");
  assert.ok(AV1, "an AV1 sample is declared");
  // Preference order, which is the order the page tries them in: the original sample first, so a
  // browser that could already decode it sees exactly what it saw before.
  assert.deepEqual(
    SAMPLES.map((sample) => sample.file),
    [HEVC.file, AV1.file],
  );
  for (const sample of SAMPLES) {
    assert.match(sample.file, /^\.\/[\w.]+\.mp4$/);
    assert.ok(sample.width > 0 && sample.height > 0, `${sample.file} declares a coded size`);
    assert.ok(sample.description.length > 0, `${sample.file} names itself`);
  }
});

test("a sample is asked about with the codec string and size it declares", () => {
  assert.deepEqual(decoderConfig(AV1), {
    codec: AV1.codec,
    codedWidth: AV1.width,
    codedHeight: AV1.height,
  });
  // Nothing in the query needs the file, which is what lets the page ask before it fetches.
  assert.deepEqual(Object.keys(decoderConfig(HEVC)).sort(), ["codec", "codedHeight", "codedWidth"]);
});

test("the browser's answer picks the sample, in preference order", () => {
  const supports = (...codecs) => (sample) => codecs.includes(sample.codec);

  // A browser with both takes the first, which is the 1080p HEVC one.
  assert.deepEqual(supportedSamples(SAMPLES, supports(HEVC.codec, AV1.codec)), [HEVC, AV1]);
  // A stock Chrome has no HEVC decoder, and this is the case issue #441 is about: it gets the
  // AV1 sample rather than the synthetic gradient.
  assert.deepEqual(supportedSamples(SAMPLES, supports(AV1.codec)), [AV1]);
  assert.deepEqual(supportedSamples(SAMPLES, supports(HEVC.codec)), [HEVC]);
});

test("a browser that reports no decoder is left with nothing to fetch", () => {
  // The page draws its synthetic gradient in this case. Picking a sample anyway would only cost
  // it a download it cannot decode.
  assert.deepEqual(supportedSamples(SAMPLES, () => false), []);
  // `isConfigSupported` resolves with `{supported: false}` rather than rejecting, and a
  // missing answer is not a yes either.
  assert.deepEqual(supportedSamples(SAMPLES, () => undefined), []);
});
