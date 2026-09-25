"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { decideAction, getHistory, sendMessage } from "@/lib/agent";
import type { ActionResource, ActionSummary, AgentResponse, ApiError, Membership } from "@/lib/types";

import ActivitySummary from "./ActivitySummary";
import AnswerText from "./AnswerText";
import ApprovalCard, { type ActionView } from "./ApprovalCard";
import ExecutionTrace from "./trace/ExecutionTrace";

type Item = {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: AgentResponse;
  action?: ActionView;
};

export function toView(a: ActionSummary | ActionResource): ActionView {
  return {
    id: a.id,
    action_type: a.action_type,
    status: a.status,
    summary: a.summary,
    arguments: a.arguments,
    arguments_hash: a.arguments_hash,
    evidence: a.evidence.map((e) => (typeof e === "string" ? e : e.citation)),
    expires_at: a.expires_at,
    result: a.result,
    failure_code: a.failure_code,
  };
}

const FRIENDLY: Record<string, string> = {
  agent_approval_pending: "Decide on the pending approval before sending another message.",
  unreachable: "The API is unreachable. Check your connection and retry.",
  timeout: "The assistant took too long to answer. Retry in a moment.",
  auth_required: "Your session has expired. Sign in again.",
  auth_invalid: "Your session has expired. Sign in again.",
  tenant_forbidden: "You do not have access to this tenant.",
  public_demo_read_only: "The public demo is read-only. Sign in with a reviewer account to use actions.",
  public_demo_unavailable: "The public demo is temporarily unavailable. Please try again later.",
  public_demo_disabled: "The public demo is not enabled on this server.",
  rate_limited: "You are sending messages too quickly. Wait a moment and try again.",
};

function describe(error: ApiError): string {
  return FRIENDLY[error.code] ?? `${error.message} (${error.code})`;
}

let counter = 0;
const localId = () => `local-${++counter}`;

export default function ChatPanel({
  tenant,
  threadId,
  readOnly = false,
}: {
  tenant: Membership;
  threadId: string;
  readOnly?: boolean;
}) {
  const [items, setItems] = useState<Item[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [inlineTraceId, setInlineTraceId] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(true);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ text: string; message: string } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // The panel is keyed by tenant+thread, so it mounts fresh for every conversation.
    let cancelled = false;
    getHistory(tenant.tenant_id, threadId).then((res) => {
      if (cancelled || !res.ok) return;
      const restored: Item[] = res.data.messages.map((m) => ({ id: m.id || localId(), role: m.role, text: m.content }));
      if (res.data.pending_action) {
        restored.push({ id: localId(), role: "assistant", text: "", action: toView(res.data.pending_action) });
      }
      setItems(restored);
    });
    return () => {
      cancelled = true;
    };
  }, [tenant.tenant_id, threadId]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [items, loading]);

  const submit = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean || loading) return;
      setError(null);
      setLoading(true);
      setItems((prev) => [...prev, { id: localId(), role: "user", text: clean }]);
      setDraft("");
      const res = await sendMessage(tenant.tenant_id, threadId, clean);
      setLoading(false);
      if (!res.ok) {
        setError({ text: clean, message: describe(res.error) });
        return;
      }
      const r = res.data;
      const id = localId();
      setItems((prev) => [
        ...prev,
        { id, role: "assistant", text: r.answer, response: r, action: r.action ? toView(r.action) : undefined },
      ]);
      setSelectedId(id); // the trace panel follows the newest run
    },
    [loading, tenant.tenant_id, threadId],
  );

  const decide = useCallback(
    async (itemId: string, action: ActionView, decision: "approve" | "reject"): Promise<string | null> => {
      const res = await decideAction(tenant.tenant_id, action.id, decision, action.arguments_hash);
      if (!res.ok) return describe(res.error);
      const updated = toView(res.data.action);
      const trace = res.data.execution_trace;
      const outcomeId = localId();
      setItems((prev) => {
        const origin = prev.find((it) => it.id === itemId);
        const next = prev.map((it) => (it.id === itemId ? { ...it, action: updated } : it));
        if (!res.data.answer) return next;
        // The resumed run's trace (proposal -> approval -> execution) belongs to the outcome.
        const last = trace?.[trace.length - 1];
        const response: AgentResponse | undefined =
          origin?.response && trace
            ? {
                ...origin.response,
                answer: res.data.answer,
                execution_trace: trace,
                duration_ms: typeof last?.metadata.duration_ms === "number" ? last.metadata.duration_ms : 0,
                action: origin.response.action ? { ...origin.response.action, status: updated.status } : null,
              }
            : undefined;
        return [...next, { id: outcomeId, role: "assistant", text: res.data.answer, response }];
      });
      if (trace) setSelectedId(outcomeId);
      return null;
    },
    [tenant.tenant_id],
  );

  const selected = items.find((it) => it.id === selectedId) ?? [...items].reverse().find((it) => it.response);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className={`flex min-h-0 flex-1 ${traceOpen ? "lg:grid lg:grid-cols-[minmax(0,1fr)_22rem]" : ""}`}>
      <div className="min-w-0 flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-6" aria-live="polite" aria-busy={loading} data-testid="messages">
        <div className="hidden justify-end lg:flex">
          <button
            type="button"
            onClick={() => setTraceOpen((v) => !v)}
            aria-expanded={traceOpen}
            aria-controls="trace-panel"
            className="rounded-md px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-inset ring-slate-300 hover:bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            {traceOpen ? "Hide execution trace" : "Show execution trace"}
          </button>
        </div>
        {items.length === 0 && !loading && (
          <div className="mx-auto max-w-md rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
            Ask about orders, shipments, invoices or company policy — for example
            <em className="block pt-2 text-slate-700">“Where is SHP-1003, and what compensation applies if it is delayed?”</em>
            {readOnly && (
              <span className="mt-3 block text-xs text-slate-500">
                Public demo: read-only. Lookups and policy answers work; actions need a reviewer account.
              </span>
            )}
          </div>
        )}
        {items.map((it) =>
          it.role === "user" ? (
            <div key={it.id} className="flex justify-end">
              <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2 text-sm text-white shadow-sm sm:max-w-[70%]">
                {it.text}
              </p>
            </div>
          ) : (
            <div key={it.id} className="flex justify-start" data-testid="assistant-message">
              <div
                className={`w-full max-w-[95%] space-y-3 rounded-2xl rounded-bl-md bg-white px-4 py-3 text-sm shadow-sm ring-1 sm:max-w-[80%] ${
                  selected?.id === it.id && traceOpen ? "ring-indigo-300 lg:ring-2" : "ring-slate-200"
                }`}
              >
                {it.text && <AnswerText text={it.text} citations={it.response?.citations ?? []} />}
                {it.response && <ActivitySummary toolCalls={it.response.tool_calls} retrievals={it.response.retrievals} />}
                {it.action && (
                  <ApprovalCard
                    action={it.action}
                    canApprove={tenant.role === "approver" && !readOnly}
                    onDecide={(d) => decide(it.id, it.action!, d)}
                  />
                )}
                {it.response && (
                  <div>
                    <button
                      type="button"
                      aria-expanded={inlineTraceId === it.id}
                      onClick={() => {
                        setSelectedId(it.id);
                        setTraceOpen(true);
                        setInlineTraceId((cur) => (cur === it.id ? null : it.id));
                      }}
                      className="text-xs font-medium text-indigo-700 hover:text-indigo-900 focus:outline-none focus-visible:underline"
                    >
                      View execution trace
                    </button>
                    {inlineTraceId === it.id && (
                      <div className="mt-2 rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200 lg:hidden" data-testid="inline-trace">
                        <ExecutionTrace response={it.response} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ),
        )}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-500" role="status">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" /> Working…
          </div>
        )}
        {error && (
          <div role="alert" className="flex flex-wrap items-center gap-3 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-800 ring-1 ring-rose-200">
            <span className="flex-1">{error.message}</span>
            <button
              type="button"
              onClick={() => {
                const text = error.text;
                setItems((prev) => prev.slice(0, -1));
                void submit(text);
              }}
              className="rounded-md bg-white px-3 py-1 font-medium text-rose-800 ring-1 ring-rose-300 hover:bg-rose-100"
            >
              Retry
            </button>
          </div>
        )}
        <div ref={endRef} />
      </div>
      {traceOpen && (
        <aside
          id="trace-panel"
          aria-label="Agent Execution Trace"
          className="hidden overflow-y-auto border-l border-slate-200 bg-white px-4 py-6 lg:block"
          data-testid="trace-panel"
        >
          <h2 className="mb-3 text-sm font-semibold text-slate-900">Agent Execution Trace</h2>
          {selected ? (
            <ExecutionTrace response={selected.response} />
          ) : (
            <p className="text-sm text-slate-500">Send a message to see how the agent executes it.</p>
          )}
          <p className="mt-6 border-t border-slate-100 pt-3 text-[11px] leading-relaxed text-slate-400">
            Recorded by the backend while the run executed: graph steps, tools, retrieval, grounding and approval state.
            No prompts, model reasoning, retrieved text or SQL.
          </p>
        </aside>
      )}
      </div>
      <form
        className="border-t border-slate-200 bg-white px-4 py-3 sm:px-6"
        onSubmit={(e) => {
          e.preventDefault();
          void submit(draft);
        }}
      >
        <label htmlFor="composer" className="sr-only">
          Message
        </label>
        <div className="flex items-end gap-2">
          <textarea
            id="composer"
            rows={1}
            maxLength={4000}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submit(draft);
              }
            }}
            placeholder="Ask CommerceOps AI…"
            className="max-h-40 min-h-11 flex-1 resize-y rounded-xl border border-slate-300 px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
          />
          <button
            type="submit"
            disabled={loading || !draft.trim()}
            className="h-11 rounded-xl bg-indigo-600 px-5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
