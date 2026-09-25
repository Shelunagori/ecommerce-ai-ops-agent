"use client";

import { useState } from "react";

import { AUTHORITY, FLOWS } from "./content";

/** Selectable request walkthroughs; each is the real graph path for that request type. */
export default function FlowWalkthrough() {
  const [active, setActive] = useState(FLOWS[0].id);
  const flow = FLOWS.find((f) => f.id === active) ?? FLOWS[0];
  return (
    <div data-testid="flow-walkthrough">
      <div role="tablist" aria-label="Request flows" className="flex flex-wrap gap-2">
        {FLOWS.map((f) => (
          <button
            key={f.id}
            role="tab"
            id={`tab-${f.id}`}
            aria-selected={f.id === active}
            aria-controls={`panel-${f.id}`}
            onClick={() => setActive(f.id)}
            className={`rounded-full px-3 py-1.5 text-sm font-medium ring-1 ring-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              f.id === active ? "bg-slate-900 text-white ring-slate-900" : "bg-white text-slate-700 ring-slate-300 hover:bg-slate-50"
            }`}
          >
            {f.tab}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`panel-${flow.id}`} aria-labelledby={`tab-${flow.id}`} className="mt-4">
        <p className="text-sm text-slate-600">
          User: <span className="font-medium text-slate-900">“{flow.prompt}”</span>
        </p>
        <ol className="mt-3 space-y-1.5">
          {flow.steps.map((step, i) => {
            const a = AUTHORITY[step.authority];
            return (
              <li key={`${flow.id}-${i}`}>
                {step.boundary && (
                  <div className="my-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-amber-700" aria-hidden>
                    <span className="h-px flex-1 border-t-2 border-dashed border-amber-400" />
                    Write boundary
                    <span className="h-px flex-1 border-t-2 border-dashed border-amber-400" />
                  </div>
                )}
                <div className={`flex items-start gap-3 rounded-lg px-3 py-2 ring-1 ring-inset ${a.tone}`}>
                  <span className="mt-0.5 font-semibold" aria-label={a.label} role="img">
                    {a.mark}
                  </span>
                  <div>
                    <p className="text-sm font-semibold">{step.title}</p>
                    <p className="text-xs opacity-80">{step.note}</p>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
