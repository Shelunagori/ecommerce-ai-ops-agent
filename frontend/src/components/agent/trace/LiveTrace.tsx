"use client";

import { useEffect, useRef, useState } from "react";

import { activeStep, placeholders, visibleSteps, type LiveRun, type LiveStep } from "@/lib/liveTrace";
import type { Capability, StepStatus } from "@/lib/types";

import TechnologyBadge, { badgesFor } from "./TechnologyBadge";

/** Re-render every `ms` while `active` (display-only elapsed timers; never drives state). */
export function useNow(active: boolean, ms = 100): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [active, ms]);
  return now;
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = typeof window !== "undefined" && window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    if (!mq) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sync with the media query on mount
    setReduced(mq.matches);
    const on = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return reduced;
}

export function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

const STATUS_TEXT: Record<StepStatus, string> = {
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  rejected: "Rejected",
  waiting: "Waiting",
  skipped: "Skipped",
};

const ICON: Record<StepStatus, { glyph: string; tone: string }> = {
  running: { glyph: "", tone: "bg-amber-50 ring-amber-300" },
  completed: { glyph: "✓", tone: "bg-emerald-50 text-emerald-700 ring-emerald-300" },
  failed: { glyph: "✕", tone: "bg-rose-50 text-rose-700 ring-rose-300" },
  rejected: { glyph: "⊘", tone: "bg-slate-100 text-slate-600 ring-slate-300" },
  waiting: { glyph: "!", tone: "bg-amber-100 text-amber-800 ring-amber-400" },
  skipped: { glyph: "–", tone: "bg-white text-slate-400 ring-slate-200" },
};

const ROW: Record<StepStatus, string> = {
  running: "bg-amber-50/70 ring-1 ring-amber-200",
  completed: "",
  failed: "bg-rose-50/60 ring-1 ring-rose-200",
  rejected: "",
  waiting: "bg-amber-50/80 ring-1 ring-amber-300",
  skipped: "opacity-70",
};

function StatusIcon({ status, reduced }: { status: StepStatus; reduced: boolean }) {
  const icon = ICON[status];
  const motion = reduced ? "" : status === "completed" || status === "failed" ? "trace-icon-pop" : "";
  return (
    <span
      key={status /* re-mount on change: the pop animation plays once per transition */}
      aria-hidden
      className={`relative z-10 mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ring-1 ring-inset transition-colors duration-300 ${icon.tone} ${motion}`}
    >
      {status === "running" ? (
        <span className={`h-2 w-2 rounded-full bg-amber-500 ${reduced ? "" : "trace-dot-running"}`} data-testid="running-dot" />
      ) : (
        icon.glyph
      )}
    </span>
  );
}

function statusText(step: LiveStep): string {
  if (step.kind === "approval" && step.status === "waiting") return "Waiting for approval";
  return STATUS_TEXT[step.status];
}

export function LiveTraceStep({
  step,
  now,
  reduced,
  active,
  last,
  alert,
}: {
  step: LiveStep;
  now: number;
  reduced: boolean;
  active: boolean;
  last: boolean;
  alert: boolean;
}) {
  const approval = step.kind === "approval" && step.status === "waiting";
  const running = step.status === "running";
  // The fade-in plays only for a row that first appears already finished; a row that changes
  // state in place (running -> completed) must not re-animate its whole body.
  const [firstStatus] = useState(step.status);
  const elapsed = running && step.startedAt !== null ? Math.max(0, now - step.startedAt) : null;
  const shown = step.durationMs ?? elapsed; // backend duration replaces the live timer
  const motion = reduced
    ? ""
    : running
      ? "trace-row-running trace-shimmer"
      : step.status === "failed"
        ? "trace-row-failed"
        : step.status === firstStatus
          ? "trace-row-appear"
          : "";
  const badges = badgesFor({ kind: step.kind, metadata: step.metadata });
  const rowTone = step.status === "waiting" && !approval ? "" : ROW[step.status];
  const skippedAs = step.detail?.startsWith("Not run")
    ? "Not run"
    : step.detail?.startsWith("Not executed")
      ? "Not executed"
      : "Not used";
  return (
    <li
      className={`relative flex gap-3 rounded-lg px-2 py-2 transition-colors duration-300 ${rowTone} ${motion}`}
      data-testid="trace-step"
      data-kind={step.kind}
      data-status={step.status}
      data-active={active ? "true" : undefined}
      data-motion={reduced ? "reduced" : "full"}
      aria-current={active ? "step" : undefined}
      role={alert ? "alert" : undefined}
    >
      {!last && (
        <span
          aria-hidden
          className={`absolute bottom-[-0.5rem] left-[1.1rem] top-7 w-px ${step.status === "completed" ? "bg-emerald-200" : "bg-slate-200"}`}
        />
      )}
      <StatusIcon status={step.status} reduced={reduced} />
      <div className="min-w-0 flex-1">
        <p className={`flex items-baseline gap-2 text-sm font-medium ${step.status === "skipped" ? "text-slate-500" : approval ? "text-amber-900" : step.status === "failed" ? "text-rose-800" : "text-slate-900"}`}>
          <span className="min-w-0 flex-1 break-words">
            {approval ? step.label.toUpperCase() : step.label}
            <span className="sr-only"> — {statusText(step)}</span>
          </span>
          {step.kind !== "response" && shown !== null && (
            <span
              className={`shrink-0 text-xs font-normal tabular-nums ${running ? "text-amber-700" : "text-slate-400"}`}
              data-testid={running ? "elapsed" : "duration"}
            >
              {seconds(shown)}
            </span>
          )}
          {step.status === "skipped" && (
            <span className="shrink-0 text-xs font-normal text-slate-400">{skippedAs}</span>
          )}
        </p>
        {step.detail && step.status !== "skipped" && (
          <p className={`text-xs ${step.status === "failed" ? "text-rose-700" : "text-slate-600"}`}>{step.detail}</p>
        )}
        {step.status === "skipped" && step.detail && skippedAs === "Not executed" && (
          <p className="text-xs text-slate-400">Nothing was changed</p>
        )}
        {approval && <p className="text-xs font-medium text-amber-800">Paused — waiting for a human decision</p>}
        {badges.length > 0 && step.status !== "skipped" && (
          <div className="mt-1 flex flex-wrap gap-1">
            {badges.map((b) => (
              <TechnologyBadge key={b} label={b} />
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

function Placeholder({ cap, last }: { cap: Capability; last: boolean }) {
  return (
    <li className="relative flex gap-3 px-2 py-2 trace-row-appear" data-testid="trace-placeholder" data-kind={cap.kind} data-status="pending">
      {!last && <span aria-hidden className="absolute bottom-[-0.5rem] left-[1.1rem] top-7 w-px bg-slate-100" />}
      <span aria-hidden className="relative z-10 mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white ring-1 ring-inset ring-slate-300" />
      <p className="text-sm text-slate-400">
        {cap.label}
        <span className="sr-only"> — Not started</span>
      </p>
    </li>
  );
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function RunHeader({ run, now }: { run: LiveRun; now: number }) {
  const elapsed = Math.max(0, (run.endedAt ?? now) - run.startedAt);
  let text: string;
  if (run.phase === "sending") text = "Sending request…";
  else if (run.phase === "running") text = `Running · ${seconds(elapsed)}`;
  else if (run.phase === "waiting_approval") text = "Waiting for human approval";
  else if (run.phase === "failed") text = `Failed after ${seconds(elapsed)}`;
  else if (run.phase === "interrupted") text = "Connection lost — the run may still finish on the server";
  else if (run.summary) {
    const s = run.summary;
    const parts = [seconds(s.durationMs), plural(s.modelCalls, "model call"), plural(s.tools, "tool")];
    if (s.retrievedSources !== null) parts.push(plural(s.retrievedSources, "retrieved source"));
    if (s.citations > 0) parts.push(plural(s.citations, "citation"));
    if (s.actionStatus) parts.push(`action ${s.actionStatus.replace("_", " ")}`);
    text = parts.join(" · ");
  } else text = seconds(elapsed);
  const tone =
    run.phase === "failed"
      ? "text-rose-700"
      : run.phase === "running" || run.phase === "sending"
        ? "text-amber-700"
        : run.phase === "waiting_approval"
          ? "text-amber-800"
          : "text-slate-500";
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Agent run</p>
      <p className={`text-xs tabular-nums ${tone}`} data-testid="run-summary" data-phase={run.phase}>
        {text}
      </p>
    </div>
  );
}

function announcement(steps: LiveStep[], run: LiveRun): string {
  const last = [...steps].reverse().find((s) => s.status !== "skipped");
  if (run.phase === "completed") return "Agent run completed";
  if (run.phase === "waiting_approval") return "Agent run paused: human approval required";
  if (!last) return run.phase === "sending" ? "Sending request" : "";
  return `${statusText(last)}: ${last.label}`;
}

/**
 * The execution timeline of ONE run. Every non-gray row was reported by the backend; gray
 * rows are capabilities not used yet. Follows the active step unless the reader scrolled up.
 */
export default function LiveTrace({ run }: { run: LiveRun }) {
  const reduced = usePrefersReducedMotion();
  const ticking = run.phase === "running" || run.phase === "sending";
  const now = useNow(ticking);
  const steps = visibleSteps(run);
  const pending = placeholders(run);
  const active = activeStep(run);
  const listRef = useRef<HTMLOListElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const container = listRef.current?.closest<HTMLElement>("[data-trace-scroll]");
    if (!container) return;
    const onScroll = () => {
      pinned.current = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!pinned.current) return; // the reader scrolled up: do not steal the scroll position
    const target = listRef.current?.querySelector<HTMLElement>('[data-active="true"]') ?? listRef.current?.lastElementChild;
    (target as HTMLElement | null)?.scrollIntoView?.({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
  }, [steps.length, active?.stepId, pending.length, reduced]);

  const runFailed = run.phase === "failed";
  return (
    <div className="space-y-3" data-testid="execution-trace" data-live-run={run.runId ?? "pending"}>
      <RunHeader run={run} now={now} />
      <p className="sr-only" aria-live="polite" role="status">
        {announcement(steps, run)}
      </p>
      <ol ref={listRef} className="space-y-1" aria-label="Execution steps in order">
        {run.phase === "sending" && steps.length === 0 && (
          <>
            <li className="relative flex gap-3 rounded-lg px-2 py-2 bg-amber-50/70 ring-1 ring-amber-200" data-testid="transport-step" data-status="running">
              <StatusIcon status="running" reduced={reduced} />
              <p className="text-sm font-medium text-slate-900">
                Sending request…<span className="sr-only"> — in progress</span>
              </p>
            </li>
            <li className="relative flex gap-3 px-2 py-2" data-testid="transport-step" data-status="pending">
              <span aria-hidden className="mt-0.5 inline-flex h-5 w-5 shrink-0 rounded-full bg-white ring-1 ring-inset ring-slate-300" />
              <p className="text-sm text-slate-400">Waiting for agent events</p>
            </li>
          </>
        )}
        {steps.map((s, i) => (
          <LiveTraceStep
            // Position keys: rows only ever append, and the final trace lists the same steps in
            // the same order, so converging on it updates rows in place (no re-mount flash).
            key={`${i}-${s.kind}`}
            step={s}
            now={now}
            reduced={reduced}
            active={active?.stepId === s.stepId}
            last={i === steps.length - 1 && pending.length === 0}
            alert={runFailed && s.status === "failed"}
          />
        ))}
        {pending.map((c, i) => (
          <Placeholder key={`p-${c.kind}`} cap={c} last={i === pending.length - 1} />
        ))}
      </ol>
      {run.phase === "failed" && run.error && (
        <p className="rounded-md bg-rose-50 px-2 py-1.5 text-xs text-rose-800 ring-1 ring-rose-200" data-testid="run-error">
          The run stopped: {run.error.message}
        </p>
      )}
    </div>
  );
}
