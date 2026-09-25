import type { AgentResponse, ExecutionTraceEvent, TraceStatus } from "@/lib/types";

import TechnologyBadge, { badgesFor } from "./TechnologyBadge";

const STATUS: Record<TraceStatus, { icon: string; label: string; tone: string }> = {
  completed: { icon: "✓", label: "Completed", tone: "bg-emerald-50 text-emerald-700 ring-emerald-200" },
  waiting: { icon: "!", label: "Waiting", tone: "bg-amber-50 text-amber-800 ring-amber-300" },
  rejected: { icon: "⊘", label: "Rejected", tone: "bg-slate-100 text-slate-600 ring-slate-300" },
  failed: { icon: "✕", label: "Failed", tone: "bg-rose-50 text-rose-700 ring-rose-200" },
};

function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function RunSummary({ response }: { response: AgentResponse }) {
  const sources = response.retrievals.reduce((n, r) => n + (r.result_count ?? 0), 0);
  const parts = [
    seconds(response.duration_ms),
    plural(response.model_calls, "model call"),
    plural(response.tool_calls.length, "tool"),
  ];
  if (response.retrievals.length > 0) parts.push(plural(sources, "retrieved source"));
  if (response.citations.length > 0) parts.push(plural(response.citations.length, "citation"));
  if (response.action) parts.push(`action ${response.action.status.replace("_", " ")}`);
  return (
    <p className="text-xs text-slate-500" data-testid="run-summary">
      {parts.join(" · ")}
    </p>
  );
}

export function ExecutionTraceStep({ event }: { event: ExecutionTraceEvent }) {
  const s = STATUS[event.status];
  const approval = event.kind === "approval" && event.status === "waiting";
  const ms = typeof event.metadata.duration_ms === "number" ? event.metadata.duration_ms : null;
  return (
    <li
      className={`relative flex gap-3 pb-4 last:pb-0 ${approval ? "rounded-lg bg-amber-50/60 p-2 ring-1 ring-amber-200" : ""}`}
      data-testid="trace-step"
      data-kind={event.kind}
      data-status={event.status}
    >
      <span
        className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ring-1 ring-inset ${s.tone}`}
        aria-label={s.label}
        role="img"
      >
        {s.icon}
      </span>
      <div className="min-w-0 flex-1">
        <p className={`text-sm font-medium ${approval ? "text-amber-900" : "text-slate-900"}`}>
          {approval ? event.label.toUpperCase() : event.label}
          {event.kind !== "response" && ms !== null && (
            <span className="ml-1.5 text-xs font-normal text-slate-400">{seconds(ms)}</span>
          )}
        </p>
        {event.detail && <p className="text-xs text-slate-600">{event.detail}</p>}
        {badgesFor(event).length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {badgesFor(event).map((b) => (
              <TechnologyBadge key={b} label={b} />
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

/** The ordered execution trace of ONE run, exactly as reported by the backend. */
export default function ExecutionTrace({ response }: { response?: AgentResponse }) {
  const events = response?.execution_trace ?? [];
  if (!response || events.length === 0) {
    return (
      <p className="text-sm text-slate-500" data-testid="trace-unavailable">
        Execution trace is available for new runs.
      </p>
    );
  }
  return (
    <div className="space-y-3" data-testid="execution-trace">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Agent run</p>
        <RunSummary response={response} />
      </div>
      <ol className="space-y-0" aria-label="Execution steps in order">
        {events.map((e) => (
          <ExecutionTraceStep key={e.sequence} event={e} />
        ))}
      </ol>
    </div>
  );
}
