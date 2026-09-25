/**
 * Verification evidence shown on /review. Numbers are copied from test runs recorded in
 * docs/COMPLETION_STATUS.md ("Current production verification" and the latest phase
 * sections) — update both together.
 * Omit a number rather than guess it.
 */
export const EVIDENCE_AS_OF = "2026-09-25";

export const EVIDENCE: { label: string; value: string; note: string }[] = [
  { label: "Backend tests", value: "1,543 passed", note: "pytest on real PostgreSQL + pgvector; 25 opt-in live-model tests skipped" },
  { label: "Frontend unit tests", value: "85 passed", note: "Vitest + Testing Library: live trace, SSE parser, provider badges, auth landing, chat, approvals, /review" },
  { label: "Browser end-to-end", value: "20 passed", note: "Playwright, full suite on the current tree: real UI → real API → PostgreSQL, deterministic model, live trace transitions; dev and production builds" },
  {
    label: "Retrieval evaluation",
    value: "hit@1 1.00 vs 0.40",
    note: "Semantic vs lexical, exact chunk, on the fixed 20-case synthetic corpus (frozen baseline)",
  },
  { label: "Action trajectory eval", value: "18 cases · 12 metrics", note: "Deterministic oracle replay validates the harness (not a live-model score)" },
  { label: "Mutation testing", value: "All current suites red", note: "Security, Cloudflare/fallback/demo budget, live trace, grounding routing, tool-call ids, capability sequencing — each removed guard turns tests red; per-set counts in COMPLETION_STATUS" },
  { label: "CI", value: "4 required jobs", note: "backend · frontend · e2e · security (gitleaks, pip-audit, npm audit)" },
];
