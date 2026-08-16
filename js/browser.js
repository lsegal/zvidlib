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
  return makeError("CANCELLED", "the browser operation was cancelled");
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
