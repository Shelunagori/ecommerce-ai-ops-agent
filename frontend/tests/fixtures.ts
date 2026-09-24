import type { ActionSummary, AgentResponse, PolicyCitation } from "@/lib/types";

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
