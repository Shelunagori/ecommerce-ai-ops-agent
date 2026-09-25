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
  retriever?: string | null;
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

/** One step of the backend's ordered, safe execution trace (backend/app/agent/trace.py). */
export type TraceEventKind =
  | "request"
  | "model"
  | "commerce_tool"
  | "retrieval"
  | "grounding"
  | "action_proposal"
  | "approval"
  | "action_execution"
  | "checkpoint"
  | "response";

export type TraceStatus = "completed" | "waiting" | "rejected" | "failed";

export type ExecutionTraceEvent = {
  sequence: number;
  kind: TraceEventKind;
  label: string;
  status: TraceStatus;
  detail: string | null;
  metadata: Record<string, string | number | boolean | null>;
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
  /** Absent from older backends; empty for runs without a trace. */
  execution_trace?: ExecutionTraceEvent[];
};

export type DecisionResponse = {
  action: ActionResource;
  answer: string | null;
  execution_trace?: ExecutionTraceEvent[] | null;
};

export type Membership = {
  tenant_id: string;
  slug: string;
  name: string;
  role: "member" | "approver";
};

export type Me = {
  subject: string;
  auth_mode: "demo" | "supabase";
  memberships: Membership[];
  /** True for a verified anonymous "Try Live Demo" visitor (read-only demo tenant). */
  public_demo?: boolean;
};

export type HistoryMessage = { id: string; role: "user" | "assistant"; content: string };
export type History = {
  thread_id: string;
  messages: HistoryMessage[];
  pending_action: ActionResource | null;
};

export type ApiError = { code: string; message: string; status: number | null };

/** Live run events (backend/app/agent/events.py), streamed while a run executes. */
export type RunEventType =
  | "run_started"
  | "step_started"
  | "step_completed"
  | "step_failed"
  | "step_skipped"
  | "approval_required"
  | "approval_resolved"
  | "run_completed"
  | "run_failed";

export type StepStatus = "running" | "completed" | "failed" | "rejected" | "waiting" | "skipped";

export type Capability = { kind: TraceEventKind; label: string };

export type RunEvent = {
  type: RunEventType;
  run_id: string;
  sequence: number;
  elapsed_ms: number;
  step_id?: string;
  kind?: TraceEventKind;
  label?: string;
  status?: StepStatus;
  detail?: string;
  metadata?: Record<string, string | number | boolean | null>;
  duration_ms?: number;
  capabilities?: Capability[];
  next_steps?: Capability[];
  response?: unknown;
  error?: { code: string; message: string; status?: number };
};
