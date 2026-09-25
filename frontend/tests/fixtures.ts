import type {
  ActionSummary,
  AgentResponse,
  ExecutionTraceEvent,
  PolicyCitation,
  TraceEventKind,
  TraceStatus,
} from "@/lib/types";

export const NS = "17243d88-ed66-5445-955b-7d7572094122";
export const V2 = "policy://delayed-shipment-compensation/v2#chunk-2";

export const citation: PolicyCitation = {
  citation: V2,
  title: "Delayed Shipment Compensation Policy",
  document_key: "delayed-shipment-compensation",
  version: 2,
  section: "Delayed Shipment Compensation Policy > Compensation",
  effective_from: "2026-08-15",
  effective_to: null,
};

export const pendingAction: ActionSummary = {
  id: "0f0e0d0c-0b0a-4908-8706-050403020100",
  action_type: "cancel_order",
  status: "pending_approval",
  summary: "Cancel order ORD-1004 (currently processing). Reason: Customer request",
  arguments: { order_number: "ORD-1004", reason: "Customer request" },
  arguments_hash: "a".repeat(64),
  evidence: [],
  expires_at: "2026-09-23T12:15:00+00:00",
  result: null,
  failure_code: null,
};

export function response(over: Partial<AgentResponse> = {}): AgentResponse {
  return {
    thread_id: "t-1",
    answer: "Done.",
    prompt_version: "commerce-assistant-v3",
    model_calls: 2,
    duration_ms: 12,
    tool_calls: [],
    retrievals: [],
    citations: [],
    action: null,
    ...over,
  };
}

type Route = { method?: string; path: string | RegExp; status?: number; body: unknown };

/** fetch mock: first matching route wins; records every call. */
export function mockFetch(routes: Route[]) {
  const calls: Array<{ method: string; url: string; body: unknown; headers: Record<string, string> }> = [];
  const fn = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : undefined, headers: (init?.headers ?? {}) as Record<string, string> });
    const idx = routes.findIndex((r) => (r.method ?? "GET") === method && (typeof r.path === "string" ? url.endsWith(r.path) : r.path.test(url)));
    if (idx < 0) return new Response(JSON.stringify({ error: { code: "route_not_found", message: "no mock" } }), { status: 404 });
    const [route] = routes.splice(idx, 1);
    return new Response(JSON.stringify(route.body), { status: route.status ?? 200, headers: { "Content-Type": "application/json" } });
  };
  globalThis.fetch = fn as typeof fetch;
  return calls;
}

export function ev(
  sequence: number,
  kind: TraceEventKind,
  label: string,
  metadata: ExecutionTraceEvent["metadata"] = {},
  status: TraceStatus = "completed",
  detail: string | null = null,
): ExecutionTraceEvent {
  return { sequence, kind, label, status, detail, metadata };
}

export const TOOLS_TRACE = [
  ev(1, "request", "Request received"),
  ev(2, "model", "Agent orchestration", { call: 1, provider: "gemini" }, "completed", "Requested commerce tool: get_shipment"),
  ev(3, "commerce_tool", "Tool: get_shipment", { tool: "get_shipment", outcome: "success", duration_ms: 4.2 }, "completed", "Deterministic, tenant-scoped PostgreSQL lookup"),
  ev(4, "model", "Agent orchestration", { call: 2, provider: "gemini" }, "completed", "Composed the final answer"),
  ev(5, "response", "Response generated", { model_calls: 2, duration_ms: 1500 }),
];

export const RAG_TRACE = [
  ev(1, "request", "Request received"),
  ev(2, "model", "Agent orchestration", { call: 1, provider: "gemini" }),
  ev(3, "retrieval", "Policy retrieval", { result_count: 3, retriever: "semantic-pgvector-v1", retrieval_mode: "semantic (pgvector)", duration_ms: 20 }, "completed", "3 eligible policy sections"),
  ev(4, "model", "Agent orchestration", { call: 2, provider: "gemini" }),
  ev(5, "grounding", "Grounding validation", { citations_verified: 1 }, "completed", "1 citation checked against this run"),
  ev(6, "response", "Response generated", { model_calls: 2, duration_ms: 17500 }),
];

export const APPROVAL_TRACE = [
  ev(1, "request", "Request received"),
  ev(2, "model", "Agent orchestration", { call: 1, provider: "gemini" }),
  ev(3, "action_proposal", "Action proposal", { action_type: "cancel_order", outcome: "pending_approval" }),
  ev(4, "approval", "Human approval required", { action_type: "cancel_order" }, "waiting", "Nothing changes until an approver decides"),
  ev(5, "checkpoint", "Run paused · state checkpointed", { durable: true }),
  ev(6, "response", "Approval request returned", { model_calls: 1, duration_ms: 900 }, "waiting"),
];

export const DECIDED_TRACE = [
  ...APPROVAL_TRACE.slice(0, 3),
  ev(4, "approval", "Approved by a human", { action_status: "succeeded" }),
  ev(5, "action_execution", "Deterministic execution", { action_status: "succeeded" }, "completed", "Executed once by application code; audit event recorded"),
  ev(6, "response", "Outcome recorded", { model_calls: 1, duration_ms: 40 }),
];
