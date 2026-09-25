/**
 * A static UI illustration of the live trace states (waiting -> running -> completed). It is
 * NOT connected to any run and says so; the real timeline lives in the application.
 */
const ROWS = [
  { label: "Request received", note: "Verified tenant scope", state: "completed" },
  { label: "Agent orchestration", note: "Requested commerce tool: get_shipment", state: "completed" },
  { label: "Tool: get_shipment", note: "Deterministic, tenant-scoped PostgreSQL lookup", state: "running" },
  { label: "Policy retrieval", note: null, state: "waiting" },
  { label: "Grounding validation", note: null, state: "waiting" },
] as const;

export default function LiveTraceIllustration() {
  return (
    <figure className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200" data-testid="live-trace-illustration">
      <figcaption className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Agent execution trace</span>
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-medium text-slate-600">
          UI illustration — not a live run
        </span>
      </figcaption>
      <ol className="space-y-1" aria-label="Illustrated trace states">
        {ROWS.map((r) => (
          <li
            key={r.label}
            className={`flex gap-3 rounded-lg px-2 py-1.5 ${r.state === "running" ? "trace-row-running bg-amber-50/70 ring-1 ring-amber-200" : ""}`}
          >
            <span
              aria-hidden
              className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ring-1 ring-inset ${
                r.state === "completed"
                  ? "bg-emerald-50 text-emerald-700 ring-emerald-300"
                  : r.state === "running"
                    ? "bg-amber-50 ring-amber-300"
                    : "bg-white ring-slate-300"
              }`}
            >
              {r.state === "completed" ? "✓" : r.state === "running" ? <span className="trace-dot-running h-2 w-2 rounded-full bg-amber-500" /> : null}
            </span>
            <div>
              <p className={`text-sm ${r.state === "waiting" ? "text-slate-400" : "font-medium text-slate-900"}`}>
                {r.label}
                <span className="sr-only"> — {r.state === "completed" ? "Completed" : r.state === "running" ? "Running" : "Not started"}</span>
              </p>
              {r.note && <p className="text-xs text-slate-600">{r.note}</p>}
            </div>
          </li>
        ))}
      </ol>
    </figure>
  );
}
