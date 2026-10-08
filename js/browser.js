export class ZvidError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "ZvidError";
    this.code = code;
  }
}

export function makeError(code, message) {
  return new ZvidError(code, message);
}

function cancellationError() {
  return makeError("CANCELED", "the browser operation was canceled");
}

async function abortable(promise, signal, onAbort) {
  if (!signal) return await promise;
  if (signal.aborted) {
    if (onAbort) await onAbort();
    throw cancellationError();
  }

  let rejectCancellation;
  const cancellation = new Promise((_, reject) => {
    rejectCancellation = reject;
  });
  const abort = () => {
    if (onAbort) Promise.resolve(onAbort()).catch(() => {});
    rejectCancellation(cancellationError());
  };
  signal.addEventListener("abort", abort, { once: true });
  try {
    return await Promise.race([promise, cancellation]);
  } finally {
    signal.removeEventListener("abort", abort);
  }
}

function checkedAppend(output, chunk, maxBytes) {
  if (!(chunk instanceof Uint8Array)) {
    throw makeError("INVALID_INPUT", "a ReadableStream chunk must be a Uint8Array");
  }
  if (output.length + chunk.byteLength > maxBytes) {
    throw makeError("RESOURCE_LIMIT", "browser input exceeds maxInputBytes");
  }
  const next = new Uint8Array(output.length + chunk.byteLength);
  next.set(output);
  next.set(chunk, output.length);
  return next;
}

export async function readBrowserSource(source, maxBytes, signal) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
    throw makeError("INVALID_INPUT", "maxInputBytes must be a non-negative safe integer");
  }

  if (source instanceof Blob) {
    if (source.size > maxBytes) {
      throw makeError("RESOURCE_LIMIT", "browser input exceeds maxInputBytes");
    }
    const buffer = await abortable(source.arrayBuffer(), signal);
    return new Uint8Array(buffer).slice();
  }

  if (source instanceof ReadableStream) {
    const reader = source.getReader();
    let output = new Uint8Array();
    try {
      while (true) {
        const result = await abortable(reader.read(), signal, () => reader.cancel());
        if (result.done) return output;
        output = checkedAppend(output, result.value, maxBytes);
      }
    } finally {
      reader.releaseLock();
    }
  }

  if (source instanceof ArrayBuffer) {
    const bytes = new Uint8Array(source);
    if (bytes.byteLength > maxBytes) {
      throw makeError("RESOURCE_LIMIT", "browser input exceeds maxInputBytes");
    }
    return bytes.slice();
  }

  if (ArrayBuffer.isView(source)) {
    const bytes = new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
    if (bytes.byteLength > maxBytes) {
      throw makeError("RESOURCE_LIMIT", "browser input exceeds maxInputBytes");
    }
    return bytes.slice();
  }

  throw makeError(
    "INVALID_INPUT",
    "source must be a Blob, ReadableStream<Uint8Array>, ArrayBuffer, or typed array",
  );
}

export function makeBlob(bytes, mimeType) {
  return new Blob([new Uint8Array(bytes).slice()], { type: mimeType });
}

export function makeTestStream(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(new Uint8Array(chunk));
      controller.close();
    },
  });
}

export function makePendingStream() {
  return new ReadableStream({ pull() {} });
}

// Test-only: loads `blob` into a muted <video>, seeks it to `seekTo` seconds,
// and resolves [duration, seekable end, currentTime after the seek, videoWidth]
// once the browser has a decoded frame there. Rejects with the media error if
// the browser cannot play the file, and after ten seconds if it never settles.
export async function probeTestVideo(blob, seekTo) {
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  const url = URL.createObjectURL(blob);
  const settle = (event) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`timed out waiting for ${event}`)), 10_000);
      video.addEventListener(
        event,
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true },
      );
      video.addEventListener(
        "error",
        () => {
          clearTimeout(timer);
          reject(new Error(`video error ${video.error?.code}: ${video.error?.message}`));
        },
        { once: true },
      );
    });
  try {
    const loaded = settle("loadeddata");
    video.src = url;
    await loaded;
    const seeked = settle("seeked");
    video.currentTime = seekTo;
    await seeked;
    const seekableEnd = video.seekable.length > 0 ? video.seekable.end(video.seekable.length - 1) : 0;
    return [video.duration, seekableEnd, video.currentTime, video.videoWidth];
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

// A range source is what `OnDemandPlayback` reads an MP4 through without
// loading all of it: a URL (a string or `URL`) read with HTTP range requests,
// a `Blob` or `File` read by slicing, or an object with a numeric `size` and a
// `read(offset, length)` method resolving to a `Uint8Array` or `ArrayBuffer`.
function isUrl(source) {
  return typeof source === "string" || source instanceof URL;
}

function isRangeReader(source) {
  return source !== null && typeof source === "object" && typeof source.read === "function";
}

function rangeSourceError() {
  return makeError(
    "INVALID_INPUT",
    "source must be a URL, a Blob, or an object with a size and a read(offset, length) method",
  );
}

// The total length of `source` in bytes. A URL's comes from the
// `Content-Range` of a one-byte range request, so the server must answer
// range requests and, cross-origin, expose that header through CORS.
export async function rangeSourceSize(source) {
  if (source instanceof Blob) return source.size;
  if (isRangeReader(source)) {
    if (!Number.isSafeInteger(source.size) || source.size < 0) {
      throw makeError("INVALID_INPUT", "a range reader's size must be a non-negative safe integer");
    }
    return source.size;
  }
  if (!isUrl(source)) throw rangeSourceError();
  let response;
  try {
    response = await fetch(source, { headers: { Range: "bytes=0-0" } });
  } catch (error) {
    throw makeError("IO", `fetching ${source}: ${error?.message ?? error}`);
  }
  await response.body?.cancel();
  if (response.status !== 206) {
    throw makeError(
      response.ok ? "UNSUPPORTED" : "IO",
      response.ok
        ? `${source} does not answer HTTP range requests`
        : `fetching ${source}: HTTP ${response.status}`,
    );
  }
  const total = /\/(\d+)\s*$/.exec(response.headers.get("Content-Range") ?? "")?.[1];
  if (total === undefined) {
    throw makeError(
      "UNSUPPORTED",
      `${source} gave no total length in its Content-Range header, or did not expose it via CORS`,
    );
  }
  return Number(total);
}

// Reads up to `length` bytes of `source` from `offset`. The read suspends
// rather than blocks: it resolves once the bytes arrive.
export async function readRangeSource(source, offset, length) {
  if (source instanceof Blob) {
    return new Uint8Array(await source.slice(offset, offset + length).arrayBuffer());
  }
  if (isRangeReader(source)) {
    const bytes = await source.read(offset, length);
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    if (bytes instanceof Uint8Array) return bytes;
    throw makeError("INVALID_INPUT", "a range reader must resolve to a Uint8Array or ArrayBuffer");
  }
  if (!isUrl(source)) throw rangeSourceError();
  let response;
  try {
    response = await fetch(source, {
      headers: { Range: `bytes=${offset}-${offset + length - 1}` },
    });
  } catch (error) {
    throw makeError("IO", `fetching ${source}: ${error?.message ?? error}`);
  }
  if (response.status !== 206) {
    await response.body?.cancel();
    throw makeError(
      response.ok ? "UNSUPPORTED" : "IO",
      response.ok
        ? `${source} does not answer HTTP range requests`
        : `fetching ${source}: HTTP ${response.status}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

// The page's clock, in seconds: what times `OnDemandPlayback` of a video with
// no audio track when it is given no `AudioContext`.
export function clockSeconds() {
  return performance.now() / 1000;
}

// Test-only: a range reader over `bytes` whose every read suspends until a
// later event-loop turn, as a network read does.
export function makeSuspendingReader(bytes) {
  const data = new Uint8Array(bytes).slice();
  return {
    size: data.byteLength,
    reads: 0,
    read(offset, length) {
      this.reads += 1;
      return new Promise((resolve) =>
        setTimeout(() => resolve(data.slice(offset, offset + length)), 0),
      );
    },
  };
}

// Test-only: a `blob:` URL for `bytes`, which `fetch` reads with range
// requests just as it does an HTTP URL.
export function makeObjectUrl(bytes, mimeType) {
  return URL.createObjectURL(makeBlob(bytes, mimeType));
}
