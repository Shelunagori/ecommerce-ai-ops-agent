"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { decideAction, getHistory, sendMessage } from "@/lib/agent";
import type { ActionResource, ActionSummary, AgentResponse, ApiError, Membership } from "@/lib/types";

import ActivitySummary from "./ActivitySummary";
import AnswerText from "./AnswerText";
import ApprovalCard, { type ActionView } from "./ApprovalCard";

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
};

function describe(error: ApiError): string {
  return FRIENDLY[error.code] ?? `${error.message} (${error.code})`;
}

let counter = 0;
const localId = () => `local-${++counter}`;

export default function ChatPanel({ tenant, threadId }: { tenant: Membership; threadId: string }) {
  const [items, setItems] = useState<Item[]>([]);
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
      setItems((prev) => [
        ...prev,
        { id: localId(), role: "assistant", text: r.answer, response: r, action: r.action ? toView(r.action) : undefined },
      ]);
    },
    [loading, tenant.tenant_id, threadId],
  );

  const decide = useCallback(
    async (itemId: string, action: ActionView, decision: "approve" | "reject"): Promise<string | null> => {
      const res = await decideAction(tenant.tenant_id, action.id, decision, action.arguments_hash);
      if (!res.ok) return describe(res.error);
      const updated = toView(res.data.action);
      setItems((prev) => {
        const next = prev.map((it) => (it.id === itemId ? { ...it, action: updated } : it));
        return res.data.answer ? [...next, { id: localId(), role: "assistant", text: res.data.answer }] : next;
      });
      return null;
    },
    [tenant.tenant_id],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-6" aria-live="polite" aria-busy={loading} data-testid="messages">
        {items.length === 0 && !loading && (
          <div className="mx-auto max-w-md rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
            Ask about orders, shipments, invoices or company policy — for example
            <em className="block pt-2 text-slate-700">“Where is SHP-1003, and what compensation applies if it is delayed?”</em>
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
              <div className="w-full max-w-[95%] space-y-3 rounded-2xl rounded-bl-md bg-white px-4 py-3 text-sm shadow-sm ring-1 ring-slate-200 sm:max-w-[80%]">
                {it.text && <AnswerText text={it.text} citations={it.response?.citations ?? []} />}
                {it.response && <ActivitySummary toolCalls={it.response.tool_calls} retrievals={it.response.retrievals} />}
                {it.action && (
                  <ApprovalCard
                    action={it.action}
                    canApprove={tenant.role === "approver"}
                    onDecide={(d) => decide(it.id, it.action!, d)}
                  />
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
