/**
 * Live execution trace state for ONE run, driven only by backend run events.
 *
 * - `sending`: the request is on its way (transport state, not an agent step)
 * - every step shown as running/completed/failed/... was reported by the backend
 * - gray placeholders are the capabilities the backend said this graph HAS (run_started), or
 *   the fixed approval path (approval_required.next_steps); never a predicted order
 * - after run_completed the view converges on the authoritative final `execution_trace`
 */
import type {
  ApiError,
  Capability,
  ExecutionTraceEvent,
  RunEvent,
  StepStatus,
  TraceEventKind,
} from "./types";

export type LiveStep = {
  stepId: string;
  kind: TraceEventKind;
  label: string;
  status: StepStatus;
  detail: string | null;
  metadata: Record<string, string | number | boolean | null>;
  /** Backend-measured duration (authoritative) once the step ended. */
  durationMs: number | null;
  /** Client clock when the backend reported the start (display-only live timer). */
  startedAt: number | null;
};

export type RunPhase = "sending" | "running" | "waiting_approval" | "completed" | "failed" | "interrupted";

export type LiveRun = {
  runId: string | null;
  phase: RunPhase;
  startedAt: number;
  endedAt: number | null;
  steps: LiveStep[];
  capabilities: Capability[];
  nextSteps: Capability[];
  /** Finished steps of the paused run a decision resumes (shown before the resumed steps). */
  prefix: ExecutionTraceEvent[];
  finalTrace: ExecutionTraceEvent[] | null;
  error: ApiError | null;
  /** Totals for the summary line once the run finished. */
  summary: RunSummaryData | null;
};

export type RunSummaryData = {
  durationMs: number;
  modelCalls: number;
  tools: number;
  retrievedSources: number | null;
  citations: number;
  actionStatus: string | null;
};

export function newRun(now: number, prefix: ExecutionTraceEvent[] = []): LiveRun {
  return {
    runId: null,
    phase: "sending",
    startedAt: now,
    endedAt: null,
    steps: [],
    capabilities: [],
    nextSteps: [],
    prefix,
    finalTrace: null,
    error: null,
    summary: null,
  };
}

function stepFrom(e: RunEvent, now: number, existing?: LiveStep): LiveStep {
  return {
    stepId: e.step_id ?? `x${e.sequence}`,
    kind: e.kind ?? existing?.kind ?? "request",
    label: e.label ?? existing?.label ?? "",
    status: e.status ?? existing?.status ?? "running",
    detail: e.detail ?? null,
    metadata: e.metadata ?? {},
    durationMs: typeof e.duration_ms === "number" ? e.duration_ms : (existing?.durationMs ?? null),
    startedAt: existing?.startedAt ?? (e.type === "step_started" ? now : null),
  };
}

/** Pure reducer: apply one backend event. Events of another run are ignored. */
export function applyEvent(run: LiveRun, e: RunEvent, now: number): LiveRun {
  if (run.runId !== null && e.run_id !== run.runId) return run;
  const next: LiveRun = { ...run, runId: run.runId ?? e.run_id };
  switch (e.type) {
    case "run_started":
      return { ...next, phase: "running", capabilities: e.capabilities ?? [] };
    case "step_started":
    case "step_completed":
    case "step_failed":
    case "step_skipped":
    case "approval_resolved":
    case "approval_required": {
      const idx = next.steps.findIndex((s) => s.stepId === e.step_id);
      const step = stepFrom(e, now, idx >= 0 ? next.steps[idx] : undefined);
      const steps = idx >= 0 ? next.steps.map((s, i) => (i === idx ? step : s)) : [...next.steps, step];
      return {
        ...next,
        phase: next.phase === "sending" ? "running" : next.phase,
        steps,
        nextSteps: e.type === "approval_required" ? (e.next_steps ?? []) : next.nextSteps,
      };
    }
    case "run_completed":
      return next; // the caller finalizes with the typed response (see `completeRun`)
    case "run_failed":
      return {
        ...next,
        phase: "failed",
        endedAt: now,
        error: {
          code: e.error?.code ?? "run_failed",
          message: e.error?.message ?? "The run failed.",
          status: e.error?.status ?? null,
        },
      };
    default:
      return next;
  }
}

/** Converge on the authoritative final trace (the response is the source of truth). */
export function completeRun(
  run: LiveRun,
  finalTrace: ExecutionTraceEvent[] | undefined | null,
  summary: RunSummaryData,
  now: number,
  waitingApproval: boolean,
): LiveRun {
  return {
    ...run,
    phase: waitingApproval ? "waiting_approval" : "completed",
    endedAt: now,
    finalTrace: finalTrace && finalTrace.length > 0 ? finalTrace : null,
    summary,
  };
}

export function interruptRun(run: LiveRun, error: ApiError, now: number): LiveRun {
  return { ...run, phase: "interrupted", endedAt: now, error };
}

/** A finished run from a plain (non-streamed) response: its final trace only. */
export function runFromTrace(trace: ExecutionTraceEvent[] | undefined, summary: RunSummaryData, now: number, waitingApproval: boolean): LiveRun {
  return completeRun({ ...newRun(now), phase: "running" }, trace ?? [], summary, now, waitingApproval);
}

const FINISHED: StepStatus[] = ["completed", "failed", "rejected", "waiting"];

function fromTrace(e: ExecutionTraceEvent, index: number, liveDuration: number | null): LiveStep {
  const measured = typeof e.metadata.duration_ms === "number" ? e.metadata.duration_ms : null;
  return {
    stepId: `f${index + 1}`,
    kind: e.kind,
    label: e.label,
    status: e.status,
    detail: e.detail,
    metadata: e.metadata,
    durationMs: e.kind === "response" ? null : (measured ?? liveDuration),
    startedAt: null,
  };
}

/**
 * The rows to render. While running: the paused-run prefix, then the backend's steps in
 * emission order. Once finished: the final trace (with the live-measured durations the final
 * trace does not carry, e.g. model calls), then the steps the backend marked skipped.
 */
export function visibleSteps(run: LiveRun): LiveStep[] {
  if (run.finalTrace) {
    const live = run.steps.filter((s) => FINISHED.includes(s.status));
    const offset = run.finalTrace.length - live.length; // resumed runs: prefix first
    const rows = run.finalTrace.map((e, i) => {
      const match = live[i - offset];
      const duration = match && match.kind === e.kind && match.label === e.label ? match.durationMs : null;
      return fromTrace(e, i, duration);
    });
    return [...rows, ...run.steps.filter((s) => s.status === "skipped")];
  }
  const prefix = run.prefix
    .filter((e) => e.status !== "waiting" && e.kind !== "checkpoint")
    .map((e, i) => fromTrace(e, i, null));
  return [...prefix, ...run.steps];
}

/** Gray rows for capabilities not reported yet (only while the run is still active). */
export function placeholders(run: LiveRun): Capability[] {
  if (run.phase === "waiting_approval") return run.nextSteps;
  if (run.phase !== "running" && run.phase !== "sending") return [];
  const seen = new Set(run.steps.map((s) => s.kind));
  return run.capabilities.filter((c) => !seen.has(c.kind));
}

export function activeStep(run: LiveRun): LiveStep | null {
  if (run.phase !== "running") return null;
  return [...run.steps].reverse().find((s) => s.status === "running") ?? null;
}

export function isActive(run: LiveRun | undefined): boolean {
  return Boolean(run && (run.phase === "sending" || run.phase === "running"));
}
