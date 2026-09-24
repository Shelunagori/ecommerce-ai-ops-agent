"use client";

import { useState } from "react";

import { formatEffective, segmentAnswer } from "@/lib/citations";
import type { PolicyCitation } from "@/lib/types";

function SourceCard({ source, index }: { source: PolicyCitation; index: number }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-lg border border-slate-200 bg-white text-sm" data-testid="source-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start gap-2 px-3 py-2 text-left hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        <span className="mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-indigo-600 px-1 text-xs font-semibold text-white">
          {index}
        </span>
        <span className="flex-1">
          <span className="font-medium text-slate-900">{source.title}</span>{" "}
          <span className="text-slate-500">v{source.version}</span>
          <span className="block text-xs text-slate-500">{source.section}</span>
        </span>
        <span aria-hidden className="text-slate-400">
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 border-t border-slate-100 px-3 py-2 text-xs text-slate-600">
          <dt className="font-medium">Citation</dt>
          <dd className="break-all font-mono">{source.citation}</dd>
          <dt className="font-medium">Document</dt>
          <dd>{source.document_key}</dd>
          <dt className="font-medium">Effective</dt>
          <dd>{formatEffective(source)}</dd>
        </dl>
      )}
    </li>
  );
}

/** Assistant text with inline numbered citation markers and expandable source cards. */
export default function AnswerText({ text, citations }: { text: string; citations: PolicyCitation[] }) {
  const segments = segmentAnswer(text, citations);
  const sources = new Map<string, { source: PolicyCitation; index: number }>();
  for (const s of segments) if (s.kind === "citation") sources.set(s.citation, { source: s.source, index: s.index });
  return (
    <div className="space-y-3">
      <p className="whitespace-pre-wrap leading-relaxed text-slate-800">
        {segments.map((s, i) =>
          s.kind === "text" ? (
            <span key={i}>{s.text}</span>
          ) : (
            <sup key={i}>
              <a
                href={`#source-${s.index}`}
                title={`${s.source.title} v${s.source.version} — ${s.source.section}`}
                className="ml-0.5 rounded bg-indigo-50 px-1 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
                aria-label={`Source ${s.index}: ${s.source.title} version ${s.source.version}`}
              >
                {s.index}
              </a>
            </sup>
          ),
        )}
      </p>
      {sources.size > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p>
          <ol className="space-y-1.5">
            {[...sources.values()].map(({ source, index }) => (
              <div key={source.citation} id={`source-${index}`}>
                <SourceCard source={source} index={index} />
              </div>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
