/** Typed calls to the agent API. The tenant is only a SELECTOR; the backend decides access. */
import { apiRequest } from "./api";
import { authHeaders } from "./auth";
import { streamRun } from "./stream";
import type { ActionResource, AgentResponse, DecisionResponse, History, Me, RunEvent } from "./types";

export async function getMe() {
  return apiRequest<Me>("/api/me", { headers: await authHeaders() });
}

export async function sendMessage(tenantId: string, threadId: string, text: string) {
  return apiRequest<AgentResponse>("/api/agent/messages", {
    method: "POST",
    body: { text, thread_id: threadId },
    headers: await authHeaders(tenantId),
  });
}

/** Same run as `sendMessage`, with its real execution steps streamed while it runs. */
export async function streamAgentMessage(
  tenantId: string,
  threadId: string,
  text: string,
  { onEvent, signal }: { onEvent: (event: RunEvent) => void; signal?: AbortSignal },
) {
  return streamRun<AgentResponse>("/api/agent/messages/stream", {
    body: { text, thread_id: threadId },
    headers: await authHeaders(tenantId),
    onEvent,
    signal,
  });
}

export async function getHistory(tenantId: string, threadId: string) {
  return apiRequest<History>(`/api/agent/threads/${encodeURIComponent(threadId)}/messages`, {
    headers: await authHeaders(tenantId),
  });
}

export async function getAction(tenantId: string, actionId: string) {
  return apiRequest<ActionResource>(`/api/agent/actions/${encodeURIComponent(actionId)}`, {
    headers: await authHeaders(tenantId),
  });
}

export async function decideAction(
  tenantId: string,
  actionId: string,
  decision: "approve" | "reject",
  argumentsHash: string,
) {
  return apiRequest<DecisionResponse>(
    `/api/agent/actions/${encodeURIComponent(actionId)}/${decision}`,
    { method: "POST", body: { arguments_hash: argumentsHash }, headers: await authHeaders(tenantId) },
  );
}

/** Same decision as `decideAction` (hash-bound, idempotent), with the resumed run streamed. */
export async function streamDecision(
  tenantId: string,
  actionId: string,
  decision: "approve" | "reject",
  argumentsHash: string,
  { onEvent, signal }: { onEvent: (event: RunEvent) => void; signal?: AbortSignal },
) {
  return streamRun<DecisionResponse>(`/api/agent/actions/${encodeURIComponent(actionId)}/${decision}/stream`, {
    body: { arguments_hash: argumentsHash },
    headers: await authHeaders(tenantId),
    onEvent,
    signal,
  });
}

export function newThreadId(): string {
  return `t-${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;
}
