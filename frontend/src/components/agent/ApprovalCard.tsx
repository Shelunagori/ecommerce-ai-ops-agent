"use client";

import { useState } from "react";

import type { ActionStatus } from "@/lib/types";

import StatusBadge from "./StatusBadge";

export type ActionView = {
  id: string;
  action_type: string;
  status: ActionStatus;
  summary: string;
  arguments: Record<string, string | null>;
  arguments_hash: string;
  evidence: string[];
  expires_at: string;
  result: Record<string, string | null> | null;
  failure_code: string | null;
};

const TITLE: Record<string, string> = {
  cancel_order: "Cancel order",
  issue_store_credit: "Issue store credit",
};

const ARG_LABEL: Record<string, string> = {
  order_number: "Order",
  customer_code: "Customer",
  amount: "Amount",
  currency: "Currency",
  reason: "Reason",
};

function formatTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/**
 * Human approval for ONE server-side action request. The approve call carries the
 * arguments hash shown here, so the server executes exactly what was displayed.
 */
export default function ApprovalCard({
  action,
  canApprove,
  onDecide,
}: {
  action: ActionView;
  canApprove: boolean;
  onDecide: (decision: "approve" | "reject") => Promise<string | null>;
}) {
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = action.status === "pending_approval";

  async function decide(decision: "approve" | "reject") {
    setBusy(decision);
    setError(null);
    const problem = await onDecide(decision);
    setBusy(null);
    if (problem) setError(problem);
  }

  return (
    <section
      aria-label={`${TITLE[action.action_type] ?? action.action_type} approval`}
      className="rounded-xl border border-amber-200 bg-amber-50/40 p-4 shadow-sm"
      data-testid="approval-card"
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold text-slate-900">{TITLE[action.action_type] ?? action.action_type}</h3>
        <StatusBadge status={action.status} />
      </header>
      <p className="mt-2 text-sm text-slate-700">{action.summary}</p>
      <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
        {Object.entries(action.arguments)
          .filter(([, v]) => v !== null && v !== "")
          .map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-slate-500">{ARG_LABEL[k] ?? k}</dt>
              <dd className="font-medium text-slate-900">{v}</dd>
            </div>
          ))}
        {action.evidence.length > 0 && (
          <>
            <dt className="text-slate-500">Policy basis</dt>
            <dd className="break-all font-mono text-xs text-slate-700">{action.evidence.join(", ")}</dd>
          </>
        )}
        {pending && (
          <>
            <dt className="text-slate-500">Expires</dt>
            <dd className="text-slate-700">{formatTime(action.expires_at)}</dd>
          </>
        )}
      </dl>
      {action.status === "failed" && (
        <p className="mt-3 text-sm text-rose-700">
          Not executed ({action.failure_code ?? "failed"}). Nothing was changed.
        </p>
      )}
      {action.status === "succeeded" && action.result && (
        <p className="mt-3 text-sm text-emerald-800">Completed.</p>
      )}
      {pending && (
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => decide("approve")}
            disabled={!canApprove || busy !== null}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy === "approve" ? "Approving…" : "Approve"}
          </button>
          <button
            type="button"
            onClick={() => decide("reject")}
            disabled={!canApprove || busy !== null}
            className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy === "reject" ? "Rejecting…" : "Reject"}
          </button>
          {!canApprove && (
            <p className="w-full text-xs text-slate-500">Only approvers of this tenant can decide.</p>
          )}
        </div>
      )}
      {error && (
        <p role="alert" className="mt-3 text-sm text-rose-700">
          {error}
        </p>
      )}
      <p className="mt-3 font-mono text-[11px] text-slate-400">
        {action.id} · {action.arguments_hash.slice(0, 12)}
      </p>
    </section>
  );
}
