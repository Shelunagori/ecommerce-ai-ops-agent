/**
 * Content for the public /review page. Every claim here is backed by code in this repository
 * (paths in comments) or by a measurement recorded in docs/COMPLETION_STATUS.md.
 */

export const REPO_URL = "https://github.com/Shelunagori/ecommerce-ai-ops-agent";

export type Authority = "deterministic" | "model" | "knowledge" | "human";

export const AUTHORITY: Record<Authority, { mark: string; label: string; tone: string }> = {
  deterministic: { mark: "◆", label: "Deterministic code / PostgreSQL", tone: "text-slate-900 bg-slate-100 ring-slate-300" },
  model: { mark: "◇", label: "Language model (proposes, never decides)", tone: "text-violet-800 bg-violet-50 ring-violet-200" },
  knowledge: { mark: "▣", label: "Policy knowledge (RAG)", tone: "text-sky-800 bg-sky-50 ring-sky-200" },
  human: { mark: "●", label: "Human decision", tone: "text-amber-900 bg-amber-50 ring-amber-300" },
};

export type FlowStep = { authority: Authority; title: string; note: string; boundary?: boolean };

export type Flow = { id: string; tab: string; prompt: string; steps: FlowStep[] };

// Graph topology: backend/app/agent/graph/builder.py, routing.py, nodes.py
export const FLOWS: Flow[] = [
  {
    id: "commerce",
    tab: "Commerce query",
    prompt: "Where is SHP-1003?",
    steps: [
      { authority: "deterministic", title: "Verified principal", note: "Tenant comes from the verified JWT, never from the model" },
      { authority: "model", title: "MODEL node", note: "Chooses a tool from an explicit allowlist" },
      { authority: "deterministic", title: "get_shipment", note: "Typed LangChain tool; tenant injected at runtime" },
      { authority: "deterministic", title: "PostgreSQL", note: "Tenant-scoped query — exact business fact" },
      { authority: "model", title: "MODEL node", note: "Answers from the tool result" },
      { authority: "deterministic", title: "Response", note: "Answer + ordered execution trace" },
    ],
  },
  {
    id: "policy",
    tab: "Policy query",
    prompt: "What compensation applies to a delayed shipment?",
    steps: [
      { authority: "model", title: "MODEL node", note: "Requests search_policy_knowledge" },
      { authority: "knowledge", title: "RETRIEVE node", note: "Trusted tenant + effective date; limit enforced" },
      { authority: "knowledge", title: "pgvector", note: "Cosine search over versioned policy chunks" },
      { authority: "model", title: "MODEL node", note: "Writes an answer citing policy:// chunks" },
      { authority: "deterministic", title: "Grounding validation", note: "Citations must come from THIS turn's retrieval" },
      { authority: "deterministic", title: "Cited answer", note: "Source cards: title, version, effective dates" },
    ],
  },
  {
    id: "mixed",
    tab: "Mixed query",
    prompt: "Where is SHP-1003, and what compensation applies if it is delayed?",
    steps: [
      { authority: "model", title: "MODEL node", note: "Commerce facts first" },
      { authority: "deterministic", title: "get_shipment → PostgreSQL", note: "Tenant-scoped shipment facts returned" },
      { authority: "model", title: "MODEL node", note: "Then policy knowledge" },
      { authority: "knowledge", title: "Policy retrieval → pgvector", note: "After retrieval, no new commerce tool may run in the turn" },
      { authority: "deterministic", title: "Grounding validation", note: "Stale or invented citations are rejected" },
      { authority: "model", title: "MODEL node", note: "Composes the cited answer" },
    ],
  },
  {
    id: "hitl",
    tab: "HITL action",
    prompt: "Cancel ORD-1004",
    steps: [
      { authority: "model", title: "MODEL node", note: "Checks facts, then calls propose_cancel_order" },
      { authority: "deterministic", title: "PROPOSE node", note: "Validates rules; persists a pending request + audit event" },
      { authority: "deterministic", title: "APPROVAL interrupt", note: "Graph pauses; state saved in PostgreSQL checkpoints" },
      { authority: "human", title: "HUMAN APPROVAL REQUIRED", note: "Approver sees exact arguments + hash", boundary: true },
      { authority: "deterministic", title: "EXECUTE node", note: "Row lock, re-check, one write, idempotency key" },
      { authority: "deterministic", title: "Audit + templated outcome", note: "The model never reports the write" },
    ],
  },
];

export const SCENARIOS = [
  { title: "Deterministic commerce lookup", body: "Orders, shipments, invoices and customers come from 12 typed, tenant-scoped read tools over PostgreSQL — never from RAG." },
  { title: "Policy RAG with citations", body: "Versioned policies, effective-date filtering and pgvector retrieval; answers must cite chunks retrieved in the same turn." },
  { title: "Mixed tool + RAG", body: "One run combines exact facts with policy knowledge, in a fixed order: commerce tools, then retrieval, then the grounded answer." },
  { title: "Approval-gated write action", body: "Cancel an order or issue synthetic store credit only after a human approves; execution is deterministic and idempotent." },
];

// backend/app/agent/tools, actions/, auth/, graph/checkpoint.py, core/production.py
export const SAFETY = [
  { title: "Trusted tenant context", body: "The verified principal sets the tenant as LangGraph runtime context; tool schemas forbid extra fields." },
  { title: "The LLM never chooses the tenant", body: "A smuggled tenant_id argument is rejected and reported by name only." },
  { title: "No raw SQL, no generic tools", body: "An explicit allowlist: 12 read tools, one policy search, two schema-only action proposals." },
  { title: "Tenant-safe database", body: "Composite (tenant_id, id) foreign keys make cross-tenant rows impossible in PostgreSQL itself." },
  { title: "Writes need a human", body: "Proposals create inert pending requests; an approver role decides; the public demo has no action tools at all." },
  { title: "Idempotent execution", body: "Canonical arguments + hash, idempotency key, claim-then-apply; duplicates return the original result." },
  { title: "Audit trail", body: "Every request, decision and outcome is an audit event written in the same transaction." },
  { title: "Safe durable state", body: "PostgreSQL checkpoints with strict JSON/msgpack serialisation — no pickle fallback." },
  { title: "Production config validation", body: "The API refuses to start with demo auth, local models, wildcard CORS or missing secrets." },
];

export const RAG_STEPS = [
  "Policy documents (Markdown, versioned)",
  "Section-aware chunks",
  "Embeddings per model profile",
  "pgvector",
  "Tenant + effective-date filter",
  "Retrieval (top chunks)",
  "Citation catalog",
  "Grounded answer",
];

export const HITL_STEPS = [
  { authority: "model" as Authority, text: "LLM proposes" },
  { authority: "deterministic" as Authority, text: "Pending request persisted" },
  { authority: "deterministic" as Authority, text: "Graph interrupt" },
  { authority: "human" as Authority, text: "Human approves or rejects" },
  { authority: "deterministic" as Authority, text: "Deterministic execution" },
  { authority: "deterministic" as Authority, text: "Audit event" },
  { authority: "deterministic" as Authority, text: "Durable checkpoint" },
];

export const DECISIONS = [
  { title: "SQL tools for facts, RAG for policy", body: "Exact, changing, tenant-scoped facts are queried; prose rules are retrieved and cited." },
  { title: "Modular monolith", body: "One FastAPI service and one PostgreSQL: no fake microservices, queues or Redis without a measured need." },
  { title: "Durable checkpoints for approvals", body: "A pending approval survives restarts and can be resumed by any API instance." },
  { title: "Provider abstraction", body: "Hosted: Cloudflare Workers AI for chat/tool calling with Gemini fallback; Gemini for policy embeddings. Local development: Ollama provides both chat and embeddings. Embedding profiles are isolated by provider/model/revision, so vector spaces never mix." },
  { title: "Fallback at the model-call boundary", body: "Only a rate-limited, timed-out or unavailable model CALL is retried on the fallback, with the same history. Tools, retrieval and approved actions are never replayed; the trace names the provider that answered." },
  { title: "Explicit capability boundaries", body: "After retrieval no commerce tool runs; the model can only propose writes; the public demo gets none." },
  { title: "Honest observability", body: "The execution trace is streamed live from real graph execution (SSE over POST) — safe metadata only, never reasoning or prompts." },
];

export const DEPLOYMENT = [
  { name: "Vercel", role: "Next.js frontend" },
  { name: "Railway", role: "FastAPI container (pre-deploy: config check, migrations, checkpoints)" },
  { name: "Supabase", role: "PostgreSQL + pgvector + Auth (incl. anonymous demo sessions)" },
  { name: "Cloudflare Workers AI", role: "Primary chat model, called from the API only (synthetic data only)" },
  { name: "Gemini", role: "Chat fallback + policy embeddings (synthetic data only)" },
  { name: "GitHub Actions", role: "CI: backend, frontend, e2e, security" },
];
