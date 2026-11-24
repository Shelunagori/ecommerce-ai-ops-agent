import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import BackendWakeNotice from "@/components/BackendWakeNotice";
import { apiGet, apiRequest } from "@/lib/api";
import {
  AWAKE_SHOWN_MS,
  dismissWake,
  getWakeState,
  reportWakeSignal,
  SLOW_AFTER_MS,
  WAKE_LIMIT_MS,
} from "@/lib/backendWake";

type Handler = (url: string, init?: RequestInit) => Promise<Response>;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const never = () => new Promise<Response>(() => {});
const networkError = () => Promise.reject(new TypeError("Failed to fetch"));

/** Routes by path suffix; records every call. */
function mockFetch(routes: Record<string, Handler>) {
  const calls: Array<{ method: string; path: string }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const path = url.replace("http://api.test", "");
    calls.push({ method: init?.method ?? "GET", path });
    const handler = routes[path];
    if (!handler) throw new Error(`unmocked ${path}`);
    return handler(url, init);
  }) as typeof fetch;
  return calls;
}

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("backend wake-up (Render free tier)", () => {
  it("a fast backend never shows anything", async () => {
    mockFetch({ "/api/me": async () => json({ subject: "x" }) });
    const res = await apiRequest("/api/me");
    expect(res.ok).toBe(true);
    await vi.advanceTimersByTimeAsync(SLOW_AFTER_MS + 100);
    expect(getWakeState().status).toBe("idle");
  });

  it("a read with no answer after 2.5s starts the wake-up; its answer ends it", async () => {
    let answer!: (r: Response) => void;
    mockFetch({
      "/api/me": () => new Promise<Response>((resolve) => (answer = resolve)),
      "/health": never,
    });
    const pending = apiRequest("/api/me");
    await vi.advanceTimersByTimeAsync(SLOW_AFTER_MS - 100);
    expect(getWakeState().status).toBe("idle");
    await vi.advanceTimersByTimeAsync(200);
    expect(getWakeState().status).toBe("waking");

    answer(json({ subject: "x" }));
    expect((await pending).ok).toBe(true);
    expect(getWakeState().status).toBe("awake");
    await vi.advanceTimersByTimeAsync(AWAKE_SHOWN_MS);
    expect(getWakeState().status).toBe("idle");
  });

  it("a read that fails while asleep is retried exactly once after /health answers", async () => {
    let meCalls = 0;
    let healthCalls = 0;
    mockFetch({
      "/api/me": async () => (++meCalls === 1 ? new Response("bad gateway", { status: 502 }) : json({ subject: "x" })),
      "/health": async () => (++healthCalls < 3 ? new Response("", { status: 503 }) : json({ status: "ok" })),
    });
    const pending = apiRequest<{ subject: string }>("/api/me");
    await vi.advanceTimersByTimeAsync(0);
    expect(getWakeState().status).toBe("waking");
    await vi.advanceTimersByTimeAsync(10_000);
    const res = await pending;
    expect(res).toEqual({ ok: true, data: { subject: "x" } });
    expect(meCalls).toBe(2);
    expect(healthCalls).toBe(3);
  });

  it("writes are never retried", async () => {
    const calls = mockFetch({
      "/api/agent/actions/a/decision": networkError,
      "/health": async () => json({ status: "ok" }),
    });
    const res = await apiRequest("/api/agent/actions/a/decision", { method: "POST", body: { decision: "approve" } });
    expect(res.ok).toBe(false);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("an application error (503 with our error body) is not treated as sleep", async () => {
    mockFetch({
      "/api/x": async () => json({ error: { code: "agent_retrieval_error", message: "no" } }, 503),
    });
    const res = await apiRequest("/api/x");
    expect(res.ok).toBe(false);
    expect(getWakeState().status).toBe("idle");
  });

  it("gives up after ~2.5 minutes; dismiss releases the waiting read with its original error", async () => {
    mockFetch({ "/health": networkError, "/health/db": networkError });
    const pending = apiGet("/health/db");
    await vi.advanceTimersByTimeAsync(WAKE_LIMIT_MS + 5_000);
    expect(getWakeState().status).toBe("unreachable");
    act(() => dismissWake());
    expect(await pending).toEqual({ ok: false, status: null, data: null, error: "Unreachable" });
    expect(getWakeState().status).toBe("idle");
  });

  it("the notice counts seconds, offers Retry when unreachable, and confirms when awake", async () => {
    let healthy = false;
    mockFetch({ "/health": async () => (healthy ? json({ status: "ok" }) : new Response("", { status: 503 })) });
    render(<BackendWakeNotice />);
    expect(screen.queryByTestId("backend-wake-notice")).not.toBeInTheDocument();

    act(() => reportWakeSignal());
    expect(screen.getByText("Waking up the backend… please wait a few seconds")).toBeInTheDocument();
    expect(screen.getByText(/free demo server sleeps when idle/)).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(5_000));
    expect(screen.getByText("5s")).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(WAKE_LIMIT_MS));
    expect(screen.getByText("Backend unreachable")).toBeInTheDocument();

    healthy = true;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(screen.getByText("Backend is awake")).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(AWAKE_SHOWN_MS));
    expect(screen.queryByTestId("backend-wake-notice")).not.toBeInTheDocument();
  });
});
