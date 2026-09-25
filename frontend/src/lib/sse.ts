/**
 * Incremental parser for `text/event-stream` (the SSE wire format, WHATWG HTML §9.2).
 *
 * Network chunks do not line up with events: one chunk can hold half an event or several.
 * `push()` buffers text until a blank line ends an event, then emits it. Handles CRLF / CR /
 * LF line endings, multi-line `data:`, comments (`: keep-alive`) and a final event without
 * a trailing blank line (`end()`).
 */
export type SseMessage = { event: string; data: string; id: string | null };

export function createSseParser(onMessage: (message: SseMessage) => void) {
  let buffer = "";
  let event = "";
  let data: string[] = [];
  let id: string | null = null;

  function dispatch() {
    if (data.length > 0) onMessage({ event: event || "message", data: data.join("\n"), id });
    event = "";
    data = [];
    id = null;
  }

  function line(raw: string) {
    if (raw === "") return dispatch();
    if (raw.startsWith(":")) return; // comment / heartbeat
    const colon = raw.indexOf(":");
    const field = colon === -1 ? raw : raw.slice(0, colon);
    let value = colon === -1 ? "" : raw.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") id = value;
    // "retry" and unknown fields are ignored
  }

  return {
    push(chunk: string) {
      buffer += chunk;
      // A trailing "\r" may be the first half of "\r\n": wait for the next chunk.
      const lines = buffer.split(/\r\n|\r(?!$)|\n/);
      buffer = lines.pop() ?? "";
      for (const l of lines) line(l);
    },
    end() {
      if (buffer) line(buffer.replace(/\r$/, ""));
      buffer = "";
      dispatch();
    },
  };
}
