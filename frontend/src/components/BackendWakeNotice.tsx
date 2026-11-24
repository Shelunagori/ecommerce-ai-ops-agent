"use client";

import { useEffect, useState } from "react";

import { dismissWake, retryWake, useBackendWake, warmUpBackend } from "@/lib/backendWake";

/** Seconds since `since`, re-rendered every second while `active`. */
function useElapsedSeconds(since: number | null, active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return since === null ? 0 : Math.max(0, Math.floor((now - since) / 1000));
}

/**
 * Bottom notice for the Render free tier: the API sleeps when idle and needs a moment to start.
 * Shown only when a request is actually waiting (see src/lib/backendWake.ts); mounted once in
 * the root layout, which also sends one silent warm-up request per page load.
 */
export default function BackendWakeNotice() {
  const wake = useBackendWake();
  const seconds = useElapsedSeconds(wake.since, wake.status === "waking");

  useEffect(() => {
    warmUpBackend();
  }, []);

  if (wake.status === "idle") return null;

  const tone =
    wake.status === "awake"
      ? "bg-emerald-50/95 text-emerald-950 ring-emerald-300"
      : wake.status === "unreachable"
        ? "bg-rose-50/95 text-rose-950 ring-rose-300"
        : "bg-orange-50/95 text-slate-900 ring-orange-300";

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex justify-center px-4 pb-4">
      <div
        role="status"
        aria-live="polite"
        data-testid="backend-wake-notice"
        className={`wake-notice pointer-events-auto flex w-full max-w-xl items-center gap-3 rounded-xl px-4 py-3 shadow-xl shadow-orange-900/10 ring-1 ${tone}`}
      >
        {wake.status === "waking" && <span aria-hidden className="wake-spinner h-5 w-5 shrink-0 rounded-full border-2 border-orange-200 border-t-orange-600" />}
        {wake.status === "awake" && <span aria-hidden className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-xs text-white">✓</span>}
        {wake.status === "unreachable" && <span aria-hidden className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-rose-600 text-xs font-bold text-white">!</span>}
        <div className="min-w-0 flex-1 text-sm">
          {wake.status === "waking" && (
            <>
              <p className="font-semibold">Waking up the backend… please wait a few seconds</p>
              <p className="mt-0.5 text-xs text-slate-600">
                The free demo server sleeps when idle; the first load can take up to a minute.{" "}
                <span className="font-medium tabular-nums text-orange-700">{seconds}s</span>
              </p>
            </>
          )}
          {wake.status === "awake" && <p className="font-semibold">Backend is awake</p>}
          {wake.status === "unreachable" && (
            <>
              <p className="font-semibold">Backend unreachable</p>
              <p className="mt-0.5 text-xs text-rose-800">It did not start within a few minutes. Try again shortly.</p>
            </>
          )}
        </div>
        {wake.status === "unreachable" && (
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={retryWake}
              className="rounded-lg bg-rose-600 px-3 py-1 text-xs font-semibold text-white hover:bg-rose-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={dismissWake}
              aria-label="Dismiss"
              className="rounded-lg px-2 py-1 text-sm text-rose-700 hover:bg-rose-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/** Small header pill for pages that need the backend (the chat). */
export function BackendWakePill() {
  const wake = useBackendWake();
  if (wake.status !== "waking" && wake.status !== "unreachable") return null;
  const waking = wake.status === "waking";
  return (
    <span
      data-testid="backend-wake-pill"
      className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${
        waking ? "bg-orange-50 text-orange-800 ring-orange-200" : "bg-rose-50 text-rose-800 ring-rose-200"
      }`}
    >
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${waking ? "trace-dot-running bg-orange-500" : "bg-rose-500"}`} />
      {waking ? "Waking backend" : "Backend unreachable"}
    </span>
  );
}
