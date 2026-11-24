/**
 * Backend wake-up tracking for the Render free tier, which stops the API after ~15 minutes idle
 * and needs up to a minute or so to start again on the next request.
 *
 * The JSON client (src/lib/api.ts) reports here: a read that takes longer than SLOW_AFTER_MS, a
 * request that gets no response at all, or a bare proxy 502/503/504 (no app error body) moves the
 * state to "waking". While waking, /health is polled until it answers ("awake") or WAKE_LIMIT_MS
 * passes ("unreachable"). Reads that failed during the wake-up wait in `waitUntilAwake()` and are
 * retried once by the client; writes are never retried here.
 */
import { useSyncExternalStore } from "react";

import { API_URL } from "./api";

export type WakeStatus = "idle" | "waking" | "awake" | "unreachable";
export type WakeState = { status: WakeStatus; since: number | null };

export const SLOW_AFTER_MS = 2_500;
export const POLL_EVERY_MS = 3_000;
export const PROBE_TIMEOUT_MS = 10_000;
export const WAKE_LIMIT_MS = 150_000;
export const AWAKE_SHOWN_MS = 3_500;

const IDLE: WakeState = { status: "idle", since: null };

let state: WakeState = IDLE;
let generation = 0; // bumps on every (re)start or reset; stale poll loops stop themselves
let awakeTimer: ReturnType<typeof setTimeout> | undefined;
let warmedUp = false;
const listeners = new Set<() => void>();
let waiters: Array<(awake: boolean) => void> = [];

function set(next: WakeState) {
  state = next;
  listeners.forEach((l) => l());
}

function settleWaiters(awake: boolean) {
  const pending = waiters;
  waiters = [];
  pending.forEach((resolve) => resolve(awake));
}

async function probe(timeoutMs: number): Promise<boolean> {
  if (!API_URL) return false;
  try {
    const res = await fetch(`${API_URL}/health`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    return res.ok;
  } catch {
    return false;
  }
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function pollUntilAwake(gen: number, startedAt: number) {
  while (gen === generation) {
    const attemptAt = Date.now();
    const ok = await probe(PROBE_TIMEOUT_MS);
    if (gen !== generation) return;
    if (ok) {
      markAwake();
      return;
    }
    if (Date.now() - startedAt >= WAKE_LIMIT_MS) {
      set({ status: "unreachable", since: state.since });
      return;
    }
    // Keep a steady cadence: a probe that already waited long is retried sooner.
    await delay(Math.max(0, POLL_EVERY_MS - (Date.now() - attemptAt)));
  }
}

function startWaking() {
  clearTimeout(awakeTimer);
  const gen = ++generation;
  const startedAt = Date.now();
  set({ status: "waking", since: startedAt });
  void pollUntilAwake(gen, startedAt);
}

function markAwake() {
  if (state.status !== "waking" && state.status !== "unreachable") return;
  generation++; // stop the poll loop
  set({ status: "awake", since: Date.now() });
  settleWaiters(true);
  clearTimeout(awakeTimer);
  awakeTimer = setTimeout(() => {
    if (state.status === "awake") set(IDLE);
  }, AWAKE_SHOWN_MS);
}

/** A read has had no answer for SLOW_AFTER_MS, or a request failed like a sleeping server. */
export function reportWakeSignal() {
  if (state.status === "idle" || state.status === "awake") startWaking();
}

/** The API answered (any app response, success or app error). */
export function reportBackendResponded() {
  markAwake();
}

/**
 * Resolves true once the backend is awake (immediately if it is not waking), or false if the
 * user dismisses the "unreachable" notice. Stays pending across "unreachable" so a manual Retry
 * still lets waiting reads complete.
 */
export function waitUntilAwake(): Promise<boolean> {
  if (state.status === "idle" || state.status === "awake") return Promise.resolve(true);
  return new Promise((resolve) => waiters.push(resolve));
}

/** Manual retry from the "unreachable" notice or status chips. */
export function retryWake() {
  if (state.status === "unreachable") startWaking();
}

/** Dismiss the notice; reads still waiting give up with their original error. */
export function dismissWake() {
  generation++;
  clearTimeout(awakeTimer);
  set(IDLE);
  settleWaiters(false);
}

/**
 * Silent warm-up, once per page load: starts the server early (e.g. while someone reads /review)
 * without showing anything. If a visible wake-up is in progress, its success ends it sooner.
 */
export function warmUpBackend() {
  if (warmedUp || typeof window === "undefined") return;
  warmedUp = true;
  void probe(WAKE_LIMIT_MS).then((ok) => {
    if (ok) markAwake();
  });
}

export function getWakeState(): WakeState {
  return state;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useBackendWake(): WakeState {
  return useSyncExternalStore(subscribe, getWakeState, () => IDLE);
}

/** Test helper: back to a clean, idle store. */
export function resetBackendWake() {
  generation++;
  clearTimeout(awakeTimer);
  warmedUp = false;
  settleWaiters(false);
  state = IDLE;
  listeners.forEach((l) => l());
}
