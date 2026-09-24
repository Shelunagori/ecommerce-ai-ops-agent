/** Mirrors of the backend API contract (backend/app/api/routes/agent.py). */

export type ActionStatus =
  | "pending_approval"
  | "approved"
  | "rejected"
  | "executing"
  | "succeeded"
  | "failed"
  | "expired";

export type ToolCallSummary = {
  round: number;
  tool: string;
  arguments: Record<string, string | number | boolean | null>;
  rejected_argument_names: string[];
  outcome: string;
  error_code: string | null;
  duration_ms: number;
};

export type RetrievalSummary = {
  round: number;
  as_of: string | null;
  result_count: number;
  citations: string[];
  outcome: "success" | "no_results" | "invalid_arguments" | "error";
  error_code: string | null;
  rejected_argument_names: string[];
  duration_ms: number;
};

export type PolicyCitation = {
  citation: string;
  title: string;
  document_key: string;
  version: number;
  section: string;
  effective_from: string;
  effective_to: string | null;
};

/** Pending/finished action as returned inline with an agent answer. */
export type ActionSummary = {
  id: string;
  action_type: "cancel_order" | "issue_store_credit";
  status: ActionStatus;
  summary: string;
  arguments: Record<string, string | null>;
  arguments_hash: string;
  evidence: string[];
  expires_at: string;
  result: Record<string, string | null> | null;
  failure_code: string | null;
};

export type AuditEvent = {
  event_type: string;
  actor: string;
  at: string;
  details: Record<string, string>;
};

/** Full action resource (GET /api/agent/actions/{id}, approve/reject responses). */
export type ActionResource = {
  id: string;
  action_type: string;
  status: ActionStatus;
  summary: string;
  arguments: Record<string, string | null>;
  arguments_hash: string;
  evidence: Array<{ citation: string; title?: string; version?: number; section?: string }>;
  created_at: string;
  expires_at: string;
  decided_at: string | null;
  completed_at: string | null;
  result: Record<string, string | null> | null;
  failure_code: string | null;
  events: AuditEvent[];
};

export type AgentResponse = {
  thread_id: string;
  answer: string;
  prompt_version: string;
  model_calls: number;
  duration_ms: number;
  tool_calls: ToolCallSummary[];
  retrievals: RetrievalSummary[];
  citations: PolicyCitation[];
  action: ActionSummary | null;
};

export type DecisionResponse = { action: ActionResource; answer: string | null };

export type Membership = {
  tenant_id: string;
  slug: string;
  name: string;
  role: "member" | "approver";
};

export type Me = { subject: string; auth_mode: "demo" | "supabase"; memberships: Membership[] };

export type HistoryMessage = { id: string; role: "user" | "assistant"; content: string };
export type History = {
  thread_id: string;
  messages: HistoryMessage[];
  pending_action: ActionResource | null;
};

export type ApiError = { code: string; message: string; status: number | null };
