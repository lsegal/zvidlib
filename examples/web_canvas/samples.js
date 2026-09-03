// The video samples this page can open, and the decision of which one to open.
//
// The page used to assume one. `BigBuckBunny.mp4` carries an HEVC track, and a browser with no
// HEVC decoder - which a stock Chrome is - had `video.get(0n)` reject with `UNSUPPORTED` and fell
// through to the synthetic gradient. That took the seek preview tier down with it, because a
// preview is a decoded picture like any other: the half of `ARCHITECTURE.md` section 3.2's seek
// requirement this example exists to show was unreachable in the browser most people would open
// it in (issue #441). So the same clip is bundled a second time with an AV1 track, which every
// current browser decodes, and the page asks the browser which of them it can decode before it
// fetches either one.
//
// Each entry declares its own track's WebCodecs codec string and coded size. Nothing here reads
// the file: `tests/web_canvas_samples_match_their_files.rs` demuxes each sample and asserts this
// declaration against the string zvidlib derives from the track's own configuration box, so a
// re-encoded or replaced sample cannot quietly drift from what this file says about it.
//
// Order is preference, not capability. HEVC comes first so a browser that has always been able
// to decode the original sample keeps getting it, at its full 1080p and with nothing about that
// path changed; AV1 is what the rest of them get instead of a gradient.
export const SAMPLES = [
  {
    file: "./BigBuckBunny.mp4",
    codec: "hev1.1.60000000.L120.90",
    width: 1920,
    height: 1080,
    description: "HEVC Main 1080p",
  },
  {
    file: "./BigBuckBunny.av1.mp4",
    codec: "av01.0.04M.08.0.110",
    width: 960,
    height: 540,
    description: "AV1 Main 540p",
  },
];

// What a sample is asked about, which is also what it would be decoded with: the codec string
// and the coded size, and nothing that needs the file to have been fetched first. That is the
// whole point of asking - a browser without an HEVC decoder never downloads the HEVC copy.
export function decoderConfig(sample) {
  return { codec: sample.codec, codedWidth: sample.width, codedHeight: sample.height };
}

// The samples to try, in preference order, given what the browser said about each one. Kept pure
// - `isSupported` is the answer, not the asking - so `samples.test.js` can assert the choice
// without a browser or a decoder, the way `scrub.js` keeps the scrub walk's decisions.
//
// A browser that reports none of them leaves this empty rather than picking one anyway: the page
// has a synthetic fallback for exactly that case, and guessing would only send it a file it
// cannot use.
export function supportedSamples(samples, isSupported) {
  return samples.filter((sample) => isSupported(sample) === true);
}
