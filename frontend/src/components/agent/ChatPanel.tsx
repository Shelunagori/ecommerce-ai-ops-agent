"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { decideAction, getHistory, sendMessage, streamAgentMessage, streamDecision } from "@/lib/agent";
import {
  activeStep,
  applyEvent,
  completeRun,
  interruptRun,
  isActive,
  newRun,
  runFromTrace,
  type LiveRun,
  type RunSummaryData,
} from "@/lib/liveTrace";
import type {
  ActionResource,
  ActionSummary,
  AgentResponse,
  ApiError,
  DecisionResponse,
  ExecutionTraceEvent,
  Membership,
  RunEvent,
} from "@/lib/types";

import ActivitySummary from "./ActivitySummary";
import AnswerText from "./AnswerText";
import ApprovalCard, { type ActionView } from "./ApprovalCard";
import ExecutionTrace from "./trace/ExecutionTrace";
import LiveTrace from "./trace/LiveTrace";

type Item = {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: AgentResponse;
  action?: ActionView;
  /** Live (or finished) execution timeline of the run that produced this item. */
  run?: LiveRun;
  /** Safe notice shown in the bubble (e.g. connection lost). */
  notice?: string;
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

export function describe(error: ApiError): string {
  return FRIENDLY[error.code] ?? `${error.message} (${error.code})`;
}

const LOST =
  "The connection was lost while the agent was working. The run may still finish on the server — it was not sent again. Reload the conversation to see the result.";

function summaryOf(r: AgentResponse): RunSummaryData {
  return {
    durationMs: r.duration_ms,
    modelCalls: r.model_calls,
    tools: r.tool_calls.length,
    retrievedSources: r.retrievals.length > 0 ? r.retrievals.reduce((n, x) => n + (x.result_count ?? 0), 0) : null,
    citations: r.citations.length,
    actionStatus: r.action?.status ?? null,
  };
}

function summaryOfTrace(trace: ExecutionTraceEvent[], actionStatus: string | null): RunSummaryData {
  const last = trace[trace.length - 1]?.metadata ?? {};
  const num = (v: unknown) => (typeof v === "number" ? v : 0);
  return {
    durationMs: num(last.duration_ms),
    modelCalls: num(last.model_calls),
    tools: num(last.tool_calls),
    retrievedSources: null,
    citations: num(last.citations),
    actionStatus,
  };
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ text: string; message: string } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const itemsRef = useRef<Item[]>([]);
  const controllers = useRef(new Set<AbortController>());

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  const loadHistory = useCallback(async () => {
    const res = await getHistory(tenant.tenant_id, threadId);
    if (!res.ok) return false;
    const restored: Item[] = res.data.messages.map((m) => ({ id: m.id || localId(), role: m.role, text: m.content }));
    if (res.data.pending_action) {
      restored.push({ id: localId(), role: "assistant", text: "", action: toView(res.data.pending_action) });
    }
    setItems(restored);
    return true;
  }, [tenant.tenant_id, threadId]);

  useEffect(() => {
    // The panel is keyed by tenant+thread, so it mounts fresh for every conversation.
    let cancelled = false;
    getHistory(tenant.tenant_id, threadId).then((res) => {
      if (cancelled || !res.ok) return;
      const restored: Item[] = res.data.messages.map((m) => ({ id: m.id || localId(), role: m.role, text: m.content }));
      if (res.data.pending_action) {
        restored.push({ id: localId(), role: "assistant", text: "", action: toView(res.data.pending_action) });
      }
      setItems((current) => (current.length === 0 ? restored : current));
    });
    const live = controllers.current;
    return () => {
      cancelled = true;
      // A new conversation (or leaving the page) stops listening to earlier runs. The server
      // still finishes them; their state is durable and reloads with that conversation.
      for (const c of live) c.abort();
      live.clear();
    };
  }, [tenant.tenant_id, threadId]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [items.length, busy]);

  const patch = useCallback((id: string, fn: (it: Item) => Item) => {
    setItems((prev) => prev.map((it) => (it.id === id ? fn(it) : it)));
  }, []);

  const onEventFor = useCallback(
    (id: string) => (e: RunEvent) => patch(id, (it) => (it.run ? { ...it, run: applyEvent(it.run, e, Date.now()) } : it)),
    [patch],
  );

  const finishMessage = useCallback(
    (id: string, r: AgentResponse) => {
      const waiting = r.action?.status === "pending_approval";
      patch(id, (it) => ({
        ...it,
        text: r.answer,
        response: r,
        action: r.action ? toView(r.action) : undefined,
        run: it.run
          ? completeRun(it.run, r.execution_trace, summaryOf(r), Date.now(), waiting)
          : runFromTrace(r.execution_trace, summaryOf(r), Date.now(), waiting),
      }));
    },
    [patch],
  );

  const submit = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean || busy) return;
      setError(null);
      setBusy(true);
      const id = localId();
      setItems((prev) => [
        ...prev,
        { id: localId(), role: "user", text: clean },
        { id, role: "assistant", text: "", run: newRun(Date.now()) },
      ]);
      setDraft("");
      setSelectedId(id); // the trace panel follows the newest run, from the first moment
      setTraceOpen(true);
      const controller = new AbortController();
      controllers.current.add(controller);
      let started = false; // did the backend accept the request and start the run?
      const apply = onEventFor(id);
      const outcome = await streamAgentMessage(tenant.tenant_id, threadId, clean, {
        onEvent: (e) => {
          started = true;
          apply(e);
        },
        signal: controller.signal,
      });
      controllers.current.delete(controller);
      if (outcome.kind === "aborted" || controller.signal.aborted) return;
      setBusy(false);
      const dropPending = (message: string) => {
        setItems((prev) => prev.filter((it) => it.id !== id));
        setError({ text: clean, message });
      };
      switch (outcome.kind) {
        case "completed":
          finishMessage(id, outcome.response);
          return;
        case "unsupported": {
          // The streaming route does not exist on this server: nothing ran, so the plain
          // endpoint is used exactly once.
          setBusy(true);
          const res = await sendMessage(tenant.tenant_id, threadId, clean);
          setBusy(false);
          if (res.ok) finishMessage(id, res.data);
          else dropPending(describe(res.error));
          return;
        }
        case "failed":
          if (!started) {
            dropPending(describe(outcome.error)); // refused before the run started
            return;
          }
          // The run started and failed: keep its (red) trace visible, and offer a retry.
          setError({ text: clean, message: describe(outcome.error) });
          return;
        case "interrupted":
          patch(id, (it) => ({ ...it, notice: LOST, run: it.run ? interruptRun(it.run, outcome.error, Date.now()) : it.run }));
          return;
      }
    },
    [busy, tenant.tenant_id, threadId, onEventFor, finishMessage, patch],
  );

  const decide = useCallback(
    async (itemId: string, action: ActionView, decision: "approve" | "reject"): Promise<string | null> => {
      const origin = itemsRef.current.find((it) => it.id === itemId);
      const prefix = origin?.run?.finalTrace ?? origin?.response?.execution_trace ?? [];
      const outcomeId = localId();
      setItems((prev) => {
        const at = prev.findIndex((it) => it.id === itemId);
        const item: Item = { id: outcomeId, role: "assistant", text: "", run: newRun(Date.now(), prefix) };
        return at < 0 ? [...prev, item] : [...prev.slice(0, at + 1), item, ...prev.slice(at + 1)];
      });
      setSelectedId(outcomeId);
      setTraceOpen(true);

      const apply = (data: DecisionResponse) => {
        const updated = toView(data.action);
        const trace = data.execution_trace ?? null;
        setItems((prev) =>
          prev.flatMap((it) => {
            if (it.id === itemId) return [{ ...it, action: updated }];
            if (it.id !== outcomeId) return [it];
            if (!data.answer) return []; // no graph was resumed: only the card changes
            const summary = summaryOfTrace(trace ?? [], updated.status);
            const run = it.run ? completeRun(it.run, trace, summary, Date.now(), false) : undefined;
            return [{ ...it, text: data.answer, run }];
          }),
        );
        if (!data.answer) setSelectedId(null);
      };
      const drop = () => setItems((prev) => prev.filter((it) => it.id !== outcomeId));

      const controller = new AbortController();
      controllers.current.add(controller);
      let started = false;
      const onEvent = onEventFor(outcomeId);
      const outcome = await streamDecision(tenant.tenant_id, action.id, decision, action.arguments_hash, {
        onEvent: (e) => {
          started = true;
          onEvent(e);
        },
        signal: controller.signal,
      });
      controllers.current.delete(controller);
      switch (outcome.kind) {
        case "aborted":
          return null;
        case "completed":
          apply(outcome.response);
          return null;
        case "unsupported": {
          // No streaming route: nothing was decided yet, so the plain call is made once.
          const res = await decideAction(tenant.tenant_id, action.id, decision, action.arguments_hash);
          if (!res.ok) {
            drop();
            return describe(res.error);
          }
          apply(res.data);
          return null;
        }
        case "failed":
          if (!started) drop();
          return describe(outcome.error);
        case "interrupted":
          patch(outcomeId, (it) => ({ ...it, notice: LOST, run: it.run ? interruptRun(it.run, outcome.error, Date.now()) : it.run }));
          return "The connection was lost. The decision may have been recorded — reload the conversation to see the current state.";
      }
    },
    [tenant.tenant_id, onEventFor, patch],
  );

  const selected =
    items.find((it) => it.id === selectedId) ?? [...items].reverse().find((it) => it.run || it.response);

  function traceFor(it: Item) {
    if (it.run) return <LiveTrace run={it.run} />;
    return <ExecutionTrace response={it.response} />;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className={`flex min-h-0 flex-1 ${traceOpen ? "lg:grid lg:grid-cols-[minmax(0,1fr)_24rem]" : ""}`}>
        <div className="min-w-0 flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-6" aria-live="polite" aria-busy={busy} data-testid="messages">
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
          {items.length === 0 && !busy && (
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
          {items.map((it) => {
            if (it.role === "user") {
              return (
                <div key={it.id} className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2 text-sm text-white shadow-sm sm:max-w-[70%]">
                    {it.text}
                  </p>
                </div>
              );
            }
            const live = isActive(it.run);
            const current = it.run ? activeStep(it.run) : null;
            const hasTrace = Boolean(it.run || it.response);
            return (
              <div key={it.id} className="flex justify-start" data-testid="assistant-message" data-live={live ? "true" : undefined}>
                <div
                  className={`w-full max-w-[95%] space-y-3 rounded-2xl rounded-bl-md bg-white px-4 py-3 text-sm shadow-sm ring-1 sm:max-w-[80%] ${
                    selected?.id === it.id && traceOpen ? "ring-indigo-300 lg:ring-2" : "ring-slate-200"
                  }`}
                >
                  {live && (
                    <div role="status" className="flex items-start gap-2 text-slate-600" data-testid="working">
                      <span aria-hidden className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-amber-500 motion-safe:animate-pulse" />
                      <div>
                        <p className="font-medium text-slate-700">CommerceOps AI is working…</p>
                        <p className="text-xs text-slate-500" data-testid="current-step">
                          {current ? `Current step: ${current.label}` : "Waiting for the agent to start"}
                        </p>
                      </div>
                    </div>
                  )}
                  {it.run?.phase === "failed" && !it.text && (
                    <p className="text-sm text-rose-800" data-testid="run-failed">
                      This request could not be completed. The execution trace shows where it stopped.
                    </p>
                  )}
                  {it.text && <AnswerText text={it.text} citations={it.response?.citations ?? []} />}
                  {it.notice && (
                    <div role="alert" className="flex flex-wrap items-center gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900 ring-1 ring-amber-200">
                      <span className="flex-1">{it.notice}</span>
                      <button
                        type="button"
                        onClick={() => void loadHistory()}
                        className="rounded-md bg-white px-2 py-1 font-medium ring-1 ring-amber-300 hover:bg-amber-100"
                      >
                        Reload conversation
                      </button>
                    </div>
                  )}
                  {it.response && <ActivitySummary toolCalls={it.response.tool_calls} retrievals={it.response.retrievals} />}
                  {it.action && (
                    <ApprovalCard
                      action={it.action}
                      canApprove={tenant.role === "approver" && !readOnly}
                      onDecide={(d) => decide(it.id, it.action!, d)}
                    />
                  )}
                  {hasTrace && (
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
                        {live ? "View live execution" : "View execution trace"}
                      </button>
                      {inlineTraceId === it.id && (
                        <div className="mt-2 rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200 lg:hidden" data-testid="inline-trace">
                          {traceFor(it)}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {error && (
            <div role="alert" className="flex flex-wrap items-center gap-3 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-800 ring-1 ring-rose-200">
              <span className="flex-1">{error.message}</span>
              <button
                type="button"
                onClick={() => {
                  const text = error.text;
                  // Manual retry only (never automatic): removes the failed exchange first.
                  setItems((prev) => {
                    const lastUser = prev.map((p) => p.role).lastIndexOf("user");
                    return lastUser >= 0 && prev[lastUser].text === text ? prev.slice(0, lastUser) : prev;
                  });
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
            data-trace-scroll
          >
            <h2 className="mb-3 text-sm font-semibold text-slate-900">Agent Execution Trace</h2>
            {selected ? (
              traceFor(selected)
            ) : (
              <p className="text-sm text-slate-500">Send a message to see how the agent executes it.</p>
            )}
            <p className="mt-6 border-t border-slate-100 pt-3 text-[11px] leading-relaxed text-slate-400">
              Reported by the backend while the run executes: graph steps, tools, retrieval, grounding and approval state.
              Operational metadata only — no prompts, model reasoning, retrieved text or SQL.
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
            disabled={busy || !draft.trim()}
            className="h-11 rounded-xl bg-indigo-600 px-5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
