/**
 * Verification evidence shown on /review. Numbers are copied from test runs recorded in
 * docs/COMPLETION_STATUS.md ("Portfolio experience" section) — update both together.
 * Omit a number rather than guess it.
 */
export const EVIDENCE_AS_OF = "2026-09-25";

export const EVIDENCE: { label: string; value: string; note: string }[] = [
  { label: "Backend tests", value: "1,346 passed", note: "pytest on real PostgreSQL + pgvector; 22 opt-in live-model tests skipped" },
  { label: "Frontend unit tests", value: "51 passed", note: "Vitest + Testing Library: auth landing, trace, chat, approvals, /review" },
  { label: "Browser end-to-end", value: "16 passed", note: "Playwright: real UI → real API → PostgreSQL, deterministic model; dev and production builds" },
  {
    label: "Retrieval evaluation",
    value: "hit@1 1.00 vs 0.40",
    note: "Semantic vs lexical, exact chunk, on the fixed 20-case synthetic corpus (frozen baseline)",
  },
  { label: "Action trajectory eval", value: "18 cases · 12 metrics", note: "Deterministic oracle replay validates the harness (not a live-model score)" },
  { label: "Mutation testing", value: "29 / 29 red", note: "Re-run for this change: security (14), public demo (9), execution trace (6); earlier sets in COMPLETION_STATUS" },
  { label: "CI", value: "4 required jobs", note: "backend · frontend · e2e · security (gitleaks, pip-audit, npm audit)" },
];
