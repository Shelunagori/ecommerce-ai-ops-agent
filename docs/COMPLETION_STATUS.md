# CommerceOps AI — completion status

Source of truth between iterations of the "Remaining Project Completion Spec". Starting
point: `main` @ `93c526e` (Step 9, end-to-end policy RAG). Baseline measured at the start:
**1092 passed, 21 skipped** (full PostgreSQL suite).

Rules followed in every phase: tests first where practical, smallest coherent change,
phase tests + regressions + mutations, docs, then this file. No commits, no pushes, no
deployments. Decisions marked **(V)** are the implementer's and can be vetoed.

| Phase | Title | Status |
| --- | --- | --- |
| 1 | Human-in-the-loop write actions (1A persistence, 1B idempotency, 1C graph approval) | complete |
| 2 | Durable PostgreSQL checkpointing | complete (API wiring in Phase 6) |
| 3 | Agent/action evaluation | complete |
| 4 | Failure injection and resilience | complete |
| 5 | Observability and auditability | complete |
| 6 | Production API surface | complete |
| 7 | Trusted authentication / tenant boundary | complete (live Supabase blocked: credentials) |
| 8 | Hosted inference readiness (hosted embeddings) | complete (live Gemini blocked: credentials) |
| 9 | Frontend chat and approval UI | complete |
| 10 | API/frontend integration tests | complete |
| 11 | Deployment readiness | complete (no deployment; image build blocked in sandbox) |
| 12 | Production/security review | complete |
| 13 | Documentation and portfolio polish | complete |

---

## Phase 1 — Human-in-the-loop write actions

**Status:** complete

### Acceptance criteria
- [x] Two narrow action types only: `cancel_order`, `issue_store_credit` (no generic write,
      no generic status update).
- [x] Tenant from trusted context only; target looked up by tenant + reference.
- [x] cancel_order allowed only from draft / confirmed / processing; shipped, delivered and
      cancelled refused at proposal AND re-checked under `SELECT … FOR UPDATE` at execution.
- [x] Synthetic store-credit ledger (`store_credit_transactions`), Decimal only, positive,
      explicit currency, linked to the originating action request, idempotency key.
- [x] Persistent `action_requests` with the full lifecycle, canonical arguments + sha256,
      summary, evidence, requested_by (safe identifiers), thread key, tool call id,
      idempotency key, expiry, decision and execution metadata, failure code.
- [x] Idempotency: same proposal → same request; conflicting reuse → `idempotency_conflict`;
      duplicate execution returns the original result; concurrent duplicates → one write.
- [x] LangGraph pause/resume: MODEL → PROPOSE → APPROVAL (`interrupt`) → human decision →
      EXECUTE → END. No write before approval; reject/expiry execute nothing; approval may be
      bound to the arguments hash; tenant cannot change; intentional interrupt ≠ crash.

### Tests
- `tests/db/test_actions.py` (service, committed transactions, concurrency, constraints)
- `tests/db/test_hitl_graph.py` (graph approval flow, scripted model)
- `tests/db/test_migrations.py` (0004 round trip)

### Migrations
- `0004_action_requests` — `action_requests`, `store_credit_transactions`.

### Decisions
- **(V)** Model-visible capabilities are *proposals* (`propose_cancel_order`,
  `propose_store_credit`), schema-only like `search_policy_knowledge`; never in ToolExecutor.
- **(V)** Execution = two transactions: claim (approved→executing) then apply (business write
  + succeeded, atomic). A stale `executing` claim (> `ACTION_EXECUTION_CLAIM_TIMEOUT_SECONDS`,
  row lock obtainable) is marked `failed/execution_interrupted`, never re-executed.
- **(V)** The outcome message after approval/rejection/expiry/failure is written by the
  application (templated from the DB row), not by the model, so a write can never be
  misreported.
- **(V)** Approval also expires an *approved but not yet executed* request after `expires_at`.
- **(V)** Store credit: amount ≤ `STORE_CREDIT_MAX_AMOUNT` (default 100.00), ≤ order total,
  currency must equal the order currency (or the customer's latest order currency when no
  order is given); ≥ 1 policy citation retrieved in the current run is required.
- **(V)** At most one OPEN request per business target (partial unique index).
- **(V)** Proposals are allowed after policy retrieval (store credit needs evidence); the
  approval gate is the prompt-injection boundary for writes: injected text can at most
  create an inert pending request that a human sees.
- **(V)** `CommerceGraphAssistant` default profile stays `RAG_PROFILE` (Step-9 tests
  unchanged); production entry points use `AGENT_PROFILE` (prompt `commerce-assistant-v3`).
- **(V)** A new user message on a thread paused for approval is refused
  (`agent_approval_pending`) unless the request expired/was decided; then the pause is
  closed deterministically first.

### Results
- Tests: `test_actions.py` 42, `test_hitl_graph.py` 26, migrations +1. Full suite after the
  phase: **1161 passed, 21 skipped** (baseline 1092/21).
- Mutations (all red, all restored): H1 approval bypassed 1 · H2 execute node runs undecided
  requests 1 · H3 interrupt removed 15 · H4 expired approval executes 3 · H5 arguments
  changed after approval 3 · H6 tenant filter removed from action lookup 2 · H7 idempotent
  re-proposal lookup removed 1 · H8 duplicate execution runs twice 4 · H9 proposal
  auto-approved (injection could write) 5 · H10 resume loses pending action state 9 · H11
  execution failure reported as success 2 · H12 ledger uniqueness removed from the model 1.
- Existing expectations changed (documented): graph CLI result keys gain `action` (null);
  Step-5 CLI output keeps its Step-5 shape (RAG/action fields excluded); migration table
  set gains the two action tables.

### Known limitations
- The graph CLI (`scripts/run_graph_assistant.py`) still runs the read-only RAG profile;
  approvals go through the API (Phase 6).
- H1/H2 are caught by a single test each (service-level); the graph can only reach EXECUTE
  after a recorded human decision, so there is no second path to test.

---

## Phase 2 — Durable PostgreSQL checkpointing

**Status:** complete (production wiring into the API happens in Phase 6)

### Acceptance criteria
- [x] Official `langgraph-checkpoint-postgres==3.1.2` (`PostgresSaver`, compatible with the
      locked `langgraph-checkpoint 4.2.0`); no custom checkpoint engine.
- [x] Psycopg `ConnectionPool` with `autocommit=True`, `prepare_threshold=0`, `dict_row`
      (package requirements; also safe behind transaction poolers).
- [x] `JsonPlusSerializer(pickle_fallback=False)` — no pickle when reading checkpoints.
- [x] Tests keep `InMemorySaver`; tenant-derived `cg1-…` keys and `scope_digest` unchanged.
- [x] Approval interrupt survives a process restart (new pool + saver + assistant) and is
      resumed from another OS process (subprocess test).
- [x] Reject / expiry after restart execute nothing.
- [x] Same raw thread id in two tenants → two unrelated checkpoint threads; tenant UUIDs
      never appear in checkpoint rows or blobs.
- [x] A crashed run (open turn, no interrupt) is closed with the failure marker on the next
      message; an approval pause is never mistaken for a crash.
- [x] Retention: `delete_checkpoint_thread` + documented policy (no automatic TTL).

### Tests
- `tests/db/test_durable_checkpoint.py` (9). Full suite: **1170 passed, 21 skipped**.

### Decisions
- **(V)** Checkpoint tables are created by the package's own idempotent `setup()`
  (`python -m scripts.setup_checkpoints`), not by Alembic: the package versions its schema in
  `checkpoint_migrations`. Tests drop them afterwards so the Alembic round trip stays exact.

### Known limitations
- P6 (unbounded thread history) and P7 (raw compiled-graph bypass stores the intruding
  HumanMessage) remain open: the public runner derives keys from the trusted tenant, so P7 is
  unreachable through it, but the raw graph behaviour is unchanged.
- No automatic checkpoint TTL; pruning is an explicit operation.

---

## Phase 3 — Agent/action evaluation

**Status:** complete

- Corpus `backend/data/eval/agent_action_cases.yaml` (18 cases: pure commerce, pure RAG,
  mixed, valid/shipped/duplicate cancellation, store credit with evidence approved / rejected
  / expired / without evidence, wrong tenant, execute without approval, malformed arguments,
  retrieval failure before an action, stale citation, two prompt-injection attempts, resume
  after restart). Every case has an explicit `as_of`.
- Evaluator `backend/app/agent/action_eval.py`: records every model capability call through
  a delegating provider (refused calls included), snapshots business state before the turn,
  at the approval pause and after the decision, applies the human decision through the public
  `resume` API (optionally after a simulated restart), re-executes approved requests.
- Metrics (no LLM judge, no prose): capability choice / ordering, argument validity, tenant
  isolation, approval-required detection, approval-before-write, write idempotency, action
  outcome, forbidden-action rejection, citation validity, resume correctness, case pass.
- CI (`tests/db/test_action_eval.py`, 23 tests): scripted ORACLE replay scores **1.00 on all
  12 metrics**; three adversaries (swapped order, skipped proposal, wrong amount) are caught
  by the metric they target.
- **(V)** No live-model action evaluation script: it would mutate the development database
  (real cancellations/credits). The live measurement path is the RAG script
  (`scripts/eval_rag.py`) plus the opt-in live tests.

## Phase 4 — Failure injection and resilience

**Status:** complete

- `tests/db/test_failure_injection.py` (19 tests) — the module docstring is the table of
  failure → expected behaviour: DB unavailable/timeout (reads and proposals), LLM timeout,
  embedding timeout, profile missing, tool exception, retrieval exception, malformed tool
  call, malformed structured output, duplicate call id, duplicate idempotency key, action DB
  failure after approval, precondition changed, expired approval, repeated resume, no
  retrieval result, checkpoint persistence error (write and read). Process restart during
  approval: `test_durable_checkpoint.py`.
- New behaviour: checkpoint-store failures (SQLAlchemy / psycopg / pool timeout) escaping
  LangGraph → `agent_state_unavailable` (sanitised). An approved action whose graph resume
  failed is resumed again idempotently (no second write).
- Found and fixed: cancelling a DRAFT order violated `ck_orders_placed_at_unless_draft`
  (drafts have no `placed_at`). Migration 0004 relaxes it to
  `status IN ('draft', 'cancelled') OR placed_at IS NOT NULL` (downgrade restores the original
  as `NOT VALID`). Test: `test_cancelling_a_draft_executes`.
- Found and fixed: `StrEnum` values leaked into checkpointed state; the durable saver now runs
  in STRICT msgpack mode (`allowed_msgpack_modules=None`) and action views hold plain strings.

## Phase 5 — Observability and auditability

**Status:** complete

- Migration `0005_agent_runs_and_audit_events`: `agent_runs` (one per run/resume, best
  effort) and `audit_events` (action lifecycle, transactional with the change), both tenant
  scoped with composite FKs to `action_requests`.
- Correlation: request id (propagated by the runner to audit writers), thread key, action
  request id, tool call id. See `docs/OBSERVABILITY.md` (SLI queries, retention).
- Optional LangSmith: off by default; on only with `LANGSMITH_TRACING=true` AND an API key.
- Tests: `tests/db/test_observability.py` (9). Full suite: **1222 passed, 21 skipped**.
- **(V)** Run records are best effort (a recorder failure logs `agent run not recorded` and
  never fails the user's run); action audit is transactional.

## Phase 6 — Production API surface

**Status:** complete

- `app/api/routes/agent.py`: `POST /api/agent/messages` ({text, thread_id}; extra fields
  refused), `GET /api/agent/threads/{thread_id}/messages` (sanitised: user messages and final
  assistant messages only, plus the pending action), `GET /api/agent/actions[?status=]`,
  `GET /api/agent/actions/{id}` (with audit trail), `POST …/{id}/approve` and `…/reject`
  (body: `arguments_hash` shown on the card), `GET /api/me`.
- Tenant and user come only from the trusted principal; approvals use action RESOURCE ids;
  cross-tenant ids → 404 `action_not_found`; approve/reject are idempotent (repeat approve
  returns the same result; an approved-but-unexecuted request is safely retried; the opposite
  decision on a resolved request → 409).
- Threads are per user inside a tenant (`u<sha256(sub)[:12]>-<thread_id>`, then the runner's
  tenant-scoped `cg1-` key).
- `AssistantError` → stable envelope with mapped HTTP status (e.g. 409 approval pending, 503
  retrieval/state, 504 LLM timeout); raw exceptions never leak.
- Production runtime (`app/api/agent_runtime.py`): hosted/configured LLM, AGENT_PROFILE,
  DURABLE PostgreSQL checkpoints, ActionService, RunRecorder; built lazily; pool closed on
  shutdown.
- Tests: `tests/db/test_agent_api.py` (20).
- **(V)** No streaming: the graph pauses for approvals and ends with deterministic outcome
  messages; the UI shows a loading state instead.

## Phase 7 — Trusted authentication / tenant boundary

**Status:** complete locally; live Supabase verification blocked (needs a project + user).

- `AUTH_MODE=demo|supabase`. demo = the old `X-Tenant-ID` header, refused when
  `APP_ENV=production` (503 `auth_not_configured`).
- supabase: `Authorization: Bearer` verified per current Supabase guidance (checked Sept
  2026): JWKS at `<SUPABASE_URL>/auth/v1/.well-known/jwks.json`, RS256/ES256, `iss =
  <SUPABASE_URL>/auth/v1`, `aud = authenticated`, `exp`/`iat`/`sub` required, only `role =
  authenticated`; `alg=none` and HS256 without an explicit legacy secret are refused.
- Migration `0006_tenant_memberships`; `X-Tenant-ID` is only a selector among memberships;
  missing membership and unknown tenant look identical (403). Roles: member / approver (only
  approvers may approve or reject). `scripts/grant_membership.py` manages memberships.
- Tests: `tests/db/test_auth.py` (22) with locally generated RSA/EC keys and a locally
  served JWKS document (real `PyJWKClient` path, no network).

## Phase 8 — Hosted inference readiness

**Status:** complete locally; live Gemini check blocked (needs `GEMINI_API_KEY`).

- `GeminiEmbeddingProvider` (`app/knowledge/embeddings/gemini.py`), verified against Google's
  current docs: `gemini-embedding-2`, explicit `output_dimensionality`, documented retrieval
  prefixes (input version `policy-embedding-input-gemini-v1`), one `Content` per text,
  network-free construction, narrow retries (429/5xx/timeouts), stable error codes, inputs
  over 16 000 chars refused (the Developer API has no auto-truncate switch).
- Provenance: model digest = sha256 of the served model name + version + dimensions (derived
  identity; the API publishes no weights hash). New revision ⇒ new profile ⇒ re-materialise.
- Same validation, materialisation and retriever; profiles never mix (tested with both
  profiles materialised side by side). Ollama unchanged; frozen baselines unchanged.
- Tests: `tests/knowledge/test_gemini_embeddings.py` (16), `tests/db/test_hosted_embeddings.py`
  (2 + 1 opt-in live).
- Full suite after Phases 6–8: **1282 passed, 22 skipped**.

## Phase 9 — Frontend chat and approval UI

**Status:** complete

- Next.js App Router: `/` = agent workspace (`src/components/agent/`): sign-in (Supabase
  email/password) or demo banner, tenant selector limited to `/api/me` memberships, chat with
  history restore, loading/retry states, grounded answers with numbered citation links and
  expandable source cards, tool/retrieval activity summary, status badges, approval card
  (arguments, evidence, expiry, arguments hash; approve/reject disabled for non-approvers;
  server errors shown verbatim-safe). Former status panels moved to `/status`.
- Client never sends a tenant it was not given; the backend remains the only authority.
- Tests: vitest + Testing Library, 4 files / 24 tests (citations, ApprovalCard, ChatPanel,
  AgentApp). `npm run lint`, `npm run typecheck`, `npm run build` green.
- **(V)** No streaming (see Phase 6). **(V)** `NEXT_PUBLIC_AUTH_MODE=demo|supabase` mirrors
  the backend's `AUTH_MODE`.

## Phase 10 — API/frontend integration tests

**Status:** complete

- API level (pytest, real PostgreSQL): `test_agent_api.py` (20) and `test_auth.py` (22) cover
  chat contract, thread continuation, citations, pending → approve / reject / expired,
  duplicate approve, hash binding, 409 while pending, cross-tenant 404, validation, LLM and
  retrieval failures, JWT failures, membership-only tenants, member-cannot-approve.
- UI level (vitest, mocked fetch): 24 tests (see Phase 9).
- **Browser end-to-end (new):** `frontend/e2e/agent.spec.ts` (Playwright 1.56, 8 tests) drives
  the REAL UI against the REAL API: `backend/tests/e2e/server.py` runs the real app on the
  dedicated `*_test` database (reset + seed + policy ingestion, durable checkpoints, action
  service, lexical retrieval) with only the chat model replaced by the deterministic
  `tests/e2e/keyword_model.py`. Covers order lookup + activity, cited policy answer + source
  card, mixed shipment fact + cited policy (commerce before retrieval), cancel → approve (order changes once; duplicate approve idempotent; other tenant 404),
  reject (unchanged), store credit with policy evidence + ledger row + duplicate approve
  returning the same ledger row, per-tenant conversations
  with history restore, network failure → retry. `npm run test:e2e`.
- Mutations (all red, restored): E1 approve drops the shown hash (3 failed), E2 no history
  restore (1), E3 tool results not fed back to the model (3).
- **(V)** The harness is test code (`backend/tests/e2e`), refuses non-`_test` databases and
  `APP_ENV=production`, drops the checkpoint tables on shutdown so the pytest migration test
  stays valid; `POST /__e2e/reset` exists only on the harness app.

## Phase 11 — Deployment readiness

**Status:** complete locally. No deployment, no hosted resources, no secrets created.

- `app/core/production.py`: `configuration_problems` (setting NAMES only): DATABASE_URL,
  DEBUG=false, AUTH_MODE=supabase + https SUPABASE_URL, explicit https CORS (no `*`), hosted
  chat + embeddings with GEMINI_API_KEY, LangSmith key if tracing. The server refuses to start
  in production with any problem (lifespan fail-fast); the per-request demo-mode refusal
  (Phase 7) stays as defence in depth.
- `GET /health/ready` (`app/db/readiness.py`): config, database, migrations at head,
  checkpoint tables — stable codes only; 503 unless all ok. `/health` stays a pure liveness
  probe.
- `scripts/check_env.py`, `scripts/predeploy.py` (check_env → `alembic upgrade head` →
  `setup_checkpoints`; stops at the first failure).
- `backend/railway.json` (Dockerfile builder, pre-deploy, `/health/ready` healthcheck,
  restart policy); Dockerfile now ships `scripts/` and `data/policies/`; still non-root and
  `$PORT`. Frontend on Vercel needs no config file **(V)**.
- `.env.example` files list every setting name (placeholders only).
- `docs/DEPLOYMENT.md`: topology, env var names, session-pooler rationale, first-time runbook,
  synthetic data bootstrap from an operator machine (bulk tools refuse production), smoke
  checks, migration/rollback policy.
- Tests: `tests/test_production.py` (21), `tests/db/test_readiness.py` (5).
  Mutations P1–P7 (no fail-fast, wildcard CORS, migrations/checkpoints always ok, predeploy
  ignores failures, demo auth in prod, readiness always 200): all red, restored.
- Container contract verified without Docker (Docker Hub is blocked in the sandbox): the
  image `CMD` run as an unprivileged user with `PORT=8123` served `/health`, readiness 503
  without a DB, and `APP_ENV=production` exited at startup listing the problems.
  **Blocked:** `docker build` (registry access) — command given in DEPLOYMENT.md.
- Full suite: **1307 passed, 22 skipped**.

## Phase 12 — Production / security review

**Status:** complete. Review: `docs/SECURITY.md`.

- Gaps found and fixed (tests first, each proven by a mutation):
  - no abuse protection on the LLM endpoint → per (user, tenant) sliding-window rate limit
    before any model call (`AGENT_RATE_LIMIT_PER_MINUTE`, 429 `rate_limited` + `Retry-After`)
    **(V: in-process)**;
  - unhandled 500s were answered outside the request context (no `X-Request-ID`, request id
    `null`) → handled inside the middleware, still generic;
  - no API security headers → `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`,
    `Cache-Control: no-store` on `/api/*`;
  - public OpenAPI/docs in production → disabled when `APP_ENV=production`;
  - no frontend security headers → CSP (self + API + Supabase origins, no framing/plugins) and
    standard headers; the e2e suite fails on any CSP violation (dev and production build);
  - two test gaps found by mutations X4/X5 (provider error text in the error detail; query
    string in the request log) → tests added, both mutations now red.
- Checks: secret scan of working tree and full git history (only intentional fake test
  values), `npm audit` 0 vulnerabilities, `pip-audit` on the locked runtime deps: none known.
- **Full mutation run on the final tree: 86 mutations, 86 red** (security X1–X14 after the
  X4/X5 test fixes; Step-9 R1–R19 incl. R11b, R3a/b; Step-8 S1–S16, T1–T5, L1–L2, B1–B9;
  Step-7 K1–K14, O1–O2), each against the full suite, sources restored. Anchors updated for
  code that moved since Step 9: K5 (`app/knowledge/limits.py`), R11 (multi-line boundary
  condition). Phase-specific sets: H1–H12 (Phase 1), E1–E3 (Phase 10), P1–P7 (Phase 11),
  CSP connect-src (Phase 12, e2e hangs → red).
- Residual risks → `pending-items.md` P16–P18, P22.

## Phase 13 — Documentation and portfolio polish

**Status:** complete.

- `README.md` rewritten: capabilities, measured results with the exact retrieval wording, and
  the action-eval caveat (oracle = harness validation, not a live-model measurement),
  architecture + graph diagrams (Mermaid), quick start, test matrix, docs index, limitations.
- `docs/architecture.md`: as-built overview table and a new "Actions, approvals and the
  production path (Phases 1–12)" section with the approval sequence diagram; Step-9-era
  "not implemented" note marked historical.
- New: `docs/DEMO_SCENARIOS.md` (the spec's 5 scenarios — commerce fact, policy RAG, mixed
  shipment + policy, approval-gated cancellation, store credit with duplicate-execution
  protection — offline and live modes, all automated in the e2e suite), `docs/DEVELOPMENT.md`
  (command reference moved from the README and updated), `docs/DEPLOYMENT.md`,
  `docs/SECURITY.md`.
- `docs/pending-items.md` reviewed: P6–P12 re-checked against the code — none is fixed, all
  stay open (P6 history trimming, P7 pre-input scope check, P8/P12 small eval sets, P9 digest
  pinning, P10 abstention threshold, P11 retrieval-less policy answers). P1/P2 resolved by
  decision **(V)** (bootstrap from an operator machine with `APP_ENV=staging`). New: P13–P22.

## Final validation (this iteration)

| Check | Result |
| --- | --- |
| `ruff check` / `ruff format --check` | clean / 244 files formatted |
| `uv lock --check` | lock up to date (80 packages) |
| Backend, real PostgreSQL | **1315 passed, 22 skipped** (skips = opt-in live provider tests) |
| Backend, no database | 757 passed, 580 skipped |
| Migrations up/down | covered by the migration round-trip tests (base ↔ head, each of 0003/0004) |
| Frontend `lint` / `typecheck` / `test` / `build` | clean / clean / 28 passed / ok |
| Browser e2e (dev and production build) | 8 passed, 8 passed, no CSP violations |
| Mutations | 86/86 red on the final tree |

**Blocked (external, not locally achievable):** live Supabase Auth, live Gemini calls,
`docker build` (registry blocked in the sandbox), any deployment — by design of the spec.

## Post-completion fix (Mac run)

- `tests/test_production.py::test_readiness_without_a_database_is_not_ready` failed on the Mac
  (1 failed, 1314 passed): the readiness probe uses the process-wide engine, which reads
  `DATABASE_URL` from the environment / `backend/.env`; the test assumed none was set (true in
  the sandbox only). Test isolation bug, not a product bug — the old expectation stays; the
  test now isolates "no DATABASE_URL" with a fixture. Reproduced red with `DATABASE_URL` set,
  green after the fix with and without it; mutating the `not_configured` branch turns it red.
  Full suite **1315 passed, 22 skipped** both with and without `DATABASE_URL`.
- `npm test` failed on the Mac: the test toolchain added in Phase 9 (vitest 5, jsdom 30 →
  undici 8) needs Node ≥ 22.22.2 / 24.15, while `package.json` still declared `>=20.9` (the
  sandbox ran Node 22.22.2, so it never showed). Reproduced with Node 20.20.2 (same
  `markAsUncloneable` error). Fixed by your decision to require Node 22: `engines` =
  `^22.22.2 || ^24.15.0 || >=26.0.0`, `frontend/.nvmrc` (22) and `frontend/.npmrc`
  (`engine-strict=true`), so Node 20 now fails at `npm ci` with `EBADENGINE` instead of at test
  time. No dependency versions changed (the lock also now records the exact
  `@playwright/test` pin). On Node 22: `npm ci`, 28 unit tests, lint, typecheck, build and
  8/8 e2e green.

## Portfolio experience (public demo, execution trace, `/review`)

**Status:** complete locally. Supabase anonymous sign-ins must be enabled by an operator
(`docs/DEPLOYMENT.md`). Nothing committed or deployed by the assistant.

- **Public demo:** verified `is_anonymous: true` → one configured tenant
  (`PUBLIC_DEMO_TENANT_SLUG`, default `bluepeak-retail`) as `member`, read-only graph profile
  `commerce-assistant-v3-public-demo` (no action tools), action resources 403
  `public_demo_read_only`, per-visitor + global demo budgets; `PUBLIC_DEMO_ENABLED` off by
  default; missing demo tenant → 503. Permanent users unchanged (`tenant_memberships`).
- **Execution trace:** `app/agent/trace.py`, built by the runner from execution records;
  `AgentResponse.execution_trace`, `DecisionOut.execution_trace` (resumed runs), additive.
  `RetrievalSummary.retriever` records the retriever identity. **(V)** Response-only, not
  persisted with history (P23).
- **Frontend:** auth landing (Try Live Demo / Reviewer Sign In / Review link), demo badge,
  collapsible trace panel (desktop) and per-answer trace (mobile), public `/review` page.
- **Contract changes (documented):** result/response key sets gain `execution_trace`
  (`test_graph_assistant.RESULT_KEYS`, `test_agent_api` contract test); `/api/me` gains
  `public_demo`.
- **Tests added:** `tests/db/test_public_demo.py` (21), `tests/db/test_execution_trace.py`
  (10); frontend `SignIn`, `ExecutionTrace`, `ChatPanelTrace`, `ReviewPage`, `AgentApp`
  additions (23 new; 28 → 51); Playwright `e2e/portfolio.spec.ts` (8).
- **Results (2026-09-25):** backend **1346 passed, 22 skipped** (baseline 1315/22), no-DB run
  758 passed; `ruff`, `ruff format`, `uv lock --check`, `alembic upgrade head` + `alembic check`
  clean; frontend lint/typecheck clean, **51** unit tests, build OK; Playwright **16/16** in dev
  and production builds (no CSP violations); gitleaks (history + changed files) clean;
  pip-audit and `npm audit --omit=dev` clean; actionlint clean.
- **Mutations (all red, restored):** public demo D1–D9 (D2 first stayed green behind the
  endpoint guard → added a principal-level test), execution trace T1–T6 (T1 needed a commerce
  tool in the safety test), security X1–X14 re-run on the new tree.
- **Blocked / manual:** enable Supabase Anonymous Sign-Ins; set `PUBLIC_DEMO_ENABLED=true` on
  Railway; live Gemini/Supabase not exercised from the sandbox.

## Live execution trace (streaming)

**Status:** complete locally. Nothing committed or deployed by the assistant.

- **Transport:** `POST /api/agent/messages/stream` and
  `POST /api/agent/actions/{id}/approve|reject/stream` (`text/event-stream`, read with
  `fetch()` + `ReadableStream`; EventSource cannot send a body, bearer token or tenant
  header). Same `assistant.run` / `_decide` → `resume` as the JSON endpoints, which are
  unchanged. Auth, tenant, rate limit and capability profile are checked before streaming.
- **Events:** `app/agent/events.py` (`RunEmitter`, strict `RunEvent` schema: run_started,
  step_started/completed/failed/skipped, approval_required/resolved, run_completed/failed).
  Emitted at the real boundaries by the runner and the MODEL / TOOLS / RETRIEVE / PROPOSE /
  EXECUTE nodes; finished steps use the same builders as `execution_trace`
  (`app/agent/trace.py` refactored into per-step builders). Sink failures are swallowed.
- **Worker model:** `app/api/streaming.py` runs the synchronous run in a worker thread and
  drains an event queue; 15 s heartbeats; a disconnect never cancels a run (V) and nothing is
  retried.
- **Frontend:** `src/lib/sse.ts` (incremental parser), `src/lib/stream.ts` (outcomes:
  completed / failed / unsupported → single JSON fallback / interrupted → reload only /
  aborted), `src/lib/liveTrace.ts` (reducer, convergence on the final trace),
  `components/agent/trace/LiveTrace.tsx` (amber running pulse + live timer, green check,
  red failure with `role=alert`, gray capabilities, "Not used"/"Not run" skipped rows,
  approval waiting without spinner, auto-follow, reduced motion), `ChatPanel` wiring (panel
  opens on Send, "CommerceOps AI is working… / Current step", mobile "View live execution",
  live approve/reject). `/review` gains a "Real-time execution trace" section with a labelled
  UI illustration.
- **Contract changes (documented):** trace approval step metadata `decision` (approved /
  rejected / expired) instead of `action_status` (the same value live and final); execution
  step adds `audit_recorded: true` on success; a failed retrieval's detail is "Policy
  retrieval service was unavailable"; a proposal refused by a database error is `failed`
  (was `rejected`). The request step of the public demo says "read-only public demo (no action
  tools)". e2e: `agent.spec.ts` routes the stream endpoint for its offline test;
  `portfolio.spec.ts` ignores the new skipped rows.
- **Harness:** `tests/e2e/server.py` gains `/__e2e/pacing` (each real model call / tool /
  retrieval / action execution takes ≥ n ms) and `/__e2e/faults` (retrieval failure).
- **Tests added:** backend `tests/db/test_agent_stream.py` (30), `tests/test_run_events.py`
  (11), `tests/db/test_auth.py` (+2 stream role checks); frontend `stream.test.ts` (10),
  `liveTrace.test.ts` (4), `LiveTrace.test.tsx` (6), `ChatPanelLive.test.tsx` (8),
  `ReviewPage` (+1); Playwright `e2e/live.spec.ts` (4).
- **Results (2026-09-25, measured this iteration):** backend **1389 passed, 22 skipped**
  (baseline 1346/22), no-DB run 769 passed; `ruff`, `ruff format`, `uv lock --check`,
  `alembic upgrade head` + `alembic check` clean; frontend lint/typecheck clean, **80** unit
  tests (baseline 51), build OK; Playwright **20/20** in dev and in the production build
  (`next build && next start`), `live.spec.ts` 5× repeated 20/20; gitleaks (history + source
  dirs) clean; pip-audit and `npm audit` clean; actionlint clean.
- **Mutations (all red, restored):** live trace L1–L15 (backend) and F1–F8 (frontend; F3 first
  stayed green as a no-op and was replaced by a real "client clock instead of backend
  duration" mutation); security X1–X14 re-run on the new tree (X11 now covers both chat
  endpoints; X6 is also caught by the new stream role tests).
- **Manual / not verified here:** SSE through Railway + Vercel (P26); live Gemini/Supabase.
