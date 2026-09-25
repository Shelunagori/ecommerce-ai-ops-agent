import { describe, expect, it } from "vitest";

import { createSseParser, type SseMessage } from "@/lib/sse";
import { streamRun } from "@/lib/stream";
import type { RunEvent } from "@/lib/types";

import { controlledStream, mockFetchStreams, runEvents, sse } from "./fixtures";

function parseAll(chunks: string[]) {
  const out: SseMessage[] = [];
  const p = createSseParser((m) => out.push(m));
  chunks.forEach((c) => p.push(c));
  p.end();
  return out;
}

describe("SSE parser", () => {
  it("reassembles an event split across network chunks", () => {
    const out = parseAll(["event: step_st", "arted\nda", 'ta: {"a":', "1}\n", "\n"]);
    expect(out).toEqual([{ event: "step_started", data: '{"a":1}', id: null }]);
  });

  it("emits several events arriving in one chunk, in order", () => {
    const out = parseAll(["id: 1\ndata: one\n\nid: 2\ndata: two\n\ndata: thr", "ee\n\n"]);
    expect(out.map((m) => [m.id, m.data])).toEqual([
      ["1", "one"],
      ["2", "two"],
      [null, "three"],
    ]);
  });

  it("handles CRLF, a CR split across chunks, comments and multi-line data", () => {
    const out = parseAll([": keep-alive\r\n\r\n", "data: a\r", "\ndata: b\r\n\r\n", "data: last"]);
    expect(out.map((m) => m.data)).toEqual(["a\nb", "last"]);
  });
});

describe("streamRun", () => {
  const e = runEvents();
  const completed = e("run_completed", { response: { answer: "ok" } });

  it("ignores malformed events and events of another run, then completes", async () => {
    const other = { ...runEvents("b".repeat(32))("step_started", { step_id: "s1", kind: "model", label: "x", status: "running" }) };
    const body =
      sse(e("run_started", { capabilities: [] })) +
      "event: step_started\ndata: {not json\n\n" +
      "data: {\"type\":\"nope\",\"run_id\":\"x\",\"sequence\":1}\n\n" +
      sse(other as RunEvent) +
      sse(completed);
    mockFetchStreams([{ method: "POST", path: "/s", sse: body }]);
    const seen: RunEvent[] = [];
    const out = await streamRun<{ answer: string }>("/s", { body: {}, onEvent: (x) => seen.push(x) });
    expect(out).toEqual({ kind: "completed", response: { answer: "ok" } });
    expect(seen.map((x) => x.type)).toEqual(["run_started", "run_completed"]);
  });

  it("run_failed is a failure with the server's safe code and status", async () => {
    const f = runEvents();
    mockFetchStreams([
      { method: "POST", path: "/s", sse: sse(f("run_started"), f("run_failed", { error: { code: "agent_retrieval_error", message: "Policy knowledge could not be retrieved.", status: 503 } })) },
    ]);
    const out = await streamRun("/s", { body: {}, onEvent: () => {} });
    expect(out).toEqual({ kind: "failed", error: { code: "agent_retrieval_error", message: "Policy knowledge could not be retrieved.", status: 503 } });
  });

  it("an HTTP refusal before streaming is a failure; a missing route is 'unsupported'", async () => {
    mockFetchStreams([{ method: "POST", path: "/s", status: 429, body: { error: { code: "rate_limited", message: "Too many requests." } } }]);
    expect(await streamRun("/s", { body: {}, onEvent: () => {} })).toMatchObject({ kind: "failed", error: { code: "rate_limited", status: 429 } });
    mockFetchStreams([]);
    expect(await streamRun("/s", { body: {}, onEvent: () => {} })).toEqual({ kind: "unsupported" });
  });

  it("a connection that ends before run_completed is 'interrupted' (never resent)", async () => {
    const s = controlledStream();
    const calls = mockFetchStreams([{ method: "POST", path: "/s", sse: s }]);
    const p = streamRun("/s", { body: {}, onEvent: () => {} });
    s.push(sse(runEvents()("run_started")));
    s.error();
    expect(await p).toMatchObject({ kind: "interrupted", error: { code: "stream_interrupted" } });
    expect(calls).toHaveLength(1);
  });

  it("no response at all is a plain 'unreachable' failure (manual retry only)", async () => {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls += 1;
      throw new TypeError("Failed to fetch");
    }) as typeof fetch;
    expect(await streamRun("/s", { body: {}, onEvent: () => {} })).toEqual({
      kind: "failed",
      error: { code: "unreachable", message: "The API is unreachable.", status: null },
    });
    expect(calls).toBe(1);
  });

  it("a silent connection times out as interrupted", async () => {
    const s = controlledStream();
    mockFetchStreams([{ method: "POST", path: "/s", sse: s }]);
    const out = await streamRun("/s", { body: {}, onEvent: () => {}, idleTimeoutMs: 30 });
    expect(out).toMatchObject({ kind: "interrupted", error: { code: "stream_timeout" } });
  });

  it("the caller's abort is reported as aborted", async () => {
    const s = controlledStream();
    mockFetchStreams([{ method: "POST", path: "/s", sse: s }]);
    const ctrl = new AbortController();
    const p = streamRun("/s", { body: {}, onEvent: () => {}, signal: ctrl.signal });
    await Promise.resolve();
    ctrl.abort();
    expect(await p).toEqual({ kind: "aborted" });
  });
});
