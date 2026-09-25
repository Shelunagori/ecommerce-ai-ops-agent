/**
 * Streaming client for the live execution trace: POST + `fetch()` + `ReadableStream`.
 *
 * Why not `EventSource`: a run needs a JSON body, an `Authorization: Bearer` header and the
 * `X-Tenant-ID` selector — EventSource can only send a GET without custom headers.
 *
 * Never retries. The outcome tells the caller what happened so it can decide safely:
 * - `completed`   run_completed arrived (carries the normal JSON response body)
 * - `failed`      no response at all (unreachable), an HTTP refusal before streaming, or
 *                 run_failed — the caller may offer a MANUAL retry, never an automatic one
 * - `unsupported` the stream route does not exist (404/405 route error): NOTHING ran, so the
 *                 caller may use the non-streaming endpoint once
 * - `interrupted` the connection broke / went silent after the stream started: the run may
 *                 still finish server-side — the caller must NOT resend it, only re-read state
 * - `aborted`     the caller aborted (new conversation, unmount)
 */
import { API_URL } from "./api";
import { createSseParser } from "./sse";
import type { ApiError, RunEvent } from "./types";

export type StreamOutcome<T> =
  | { kind: "completed"; response: T }
  | { kind: "failed"; error: ApiError }
  | { kind: "unsupported" }
  | { kind: "interrupted"; error: ApiError }
  | { kind: "aborted" };

const RUN_EVENT_TYPES = new Set([
  "run_started",
  "step_started",
  "step_completed",
  "step_failed",
  "step_skipped",
  "approval_required",
  "approval_resolved",
  "run_completed",
  "run_failed",
]);

/** The server sends a heartbeat every 15 s; this long without a byte means a dead link. */
export const IDLE_TIMEOUT_MS = 45_000;

function isRunEvent(value: unknown): value is RunEvent {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.type === "string" &&
    RUN_EVENT_TYPES.has(v.type) &&
    typeof v.run_id === "string" &&
    typeof v.sequence === "number"
  );
}

export async function streamRun<T>(
  path: string,
  {
    body,
    headers = {},
    signal,
    onEvent,
    idleTimeoutMs = IDLE_TIMEOUT_MS,
  }: {
    body: unknown;
    headers?: Record<string, string>;
    signal?: AbortSignal;
    onEvent: (event: RunEvent) => void;
    idleTimeoutMs?: number;
  },
): Promise<StreamOutcome<T>> {
  if (!API_URL) {
    return { kind: "failed", error: { code: "api_not_configured", message: "NEXT_PUBLIC_API_URL is not set.", status: null } };
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort);
  let idle: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  const arm = () => {
    clearTimeout(idle);
    idle = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, idleTimeoutMs);
  };
  const interrupted = (message: string): StreamOutcome<T> => ({
    kind: "interrupted",
    error: { code: timedOut ? "stream_timeout" : "stream_interrupted", message, status: null },
  });

  try {
    arm();
    let res: Response;
    try {
      res = await fetch(`${API_URL}${path}`, {
        method: "POST",
        headers: { Accept: "text/event-stream", "Content-Type": "application/json", ...headers },
        body: JSON.stringify(body),
        cache: "no-store",
        signal: controller.signal,
      });
    } catch {
      if (signal?.aborted) return { kind: "aborted" };
      // No response at all (same as the JSON client): reported as a normal failure. The
      // caller offers only a MANUAL retry; decisions are idempotent server-side.
      return {
        kind: "failed",
        error: timedOut
          ? { code: "timeout", message: "The request timed out.", status: null }
          : { code: "unreachable", message: "The API is unreachable.", status: null },
      };
    }

    const type = res.headers.get("Content-Type") ?? "";
    if (!type.includes("text/event-stream")) {
      const data = (await res.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
      const code = data?.error?.code;
      if ((res.status === 404 || res.status === 405) && (!code || code === "route_not_found" || code === "method_not_allowed")) {
        return { kind: "unsupported" }; // the route itself is missing: nothing was executed
      }
      return {
        kind: "failed",
        error: {
          code: code ?? `http_${res.status}`,
          message: data?.error?.message ?? `Request failed (HTTP ${res.status}).`,
          status: res.status,
        },
      };
    }
    if (!res.body) return interrupted("The response had no body.");

    let runId: string | null = null;
    let outcome: StreamOutcome<T> | null = null;
    const parser = createSseParser((message) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(message.data);
      } catch {
        return; // malformed event: ignored, never fatal
      }
      if (!isRunEvent(parsed) || outcome) return;
      if (parsed.type === "run_started") runId ??= parsed.run_id;
      if (runId !== null && parsed.run_id !== runId) return; // not this run's event
      runId ??= parsed.run_id;
      onEvent(parsed);
      if (parsed.type === "run_completed") outcome = { kind: "completed", response: parsed.response as T };
      if (parsed.type === "run_failed") {
        outcome = {
          kind: "failed",
          error: {
            code: parsed.error?.code ?? "run_failed",
            message: parsed.error?.message ?? "The run failed.",
            status: parsed.error?.status ?? null,
          },
        };
      }
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        arm();
        parser.push(decoder.decode(value, { stream: true }));
        if (outcome) break;
      }
      if (!outcome) {
        parser.push(decoder.decode());
        parser.end();
      }
    } catch {
      if (signal?.aborted) return { kind: "aborted" };
      if (outcome) return outcome;
      return interrupted(timedOut ? "The agent stopped reporting progress." : "The connection was lost.");
    } finally {
      reader.cancel().catch(() => undefined);
    }
    if (outcome) return outcome;
    if (signal?.aborted) return { kind: "aborted" };
    return interrupted("The stream ended before the run finished.");
  } finally {
    clearTimeout(idle);
    signal?.removeEventListener("abort", abort);
  }
}
