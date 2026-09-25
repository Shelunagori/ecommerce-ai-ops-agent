# Security review (Phase 12)

Scope: the backend API, agent graph, action execution, auth boundary, frontend and deployment
configuration as they exist locally. All data is synthetic; nothing is deployed. Each control
below is backed by tests, and the critical ones by mutations that must turn the suite red.

## Trust boundaries

| Input | Trusted? | Where it is handled |
| --- | --- | --- |
| `Authorization: Bearer` (Supabase JWT) | only after verification | `app/auth/jwt.py`: JWKS (RS256/ES256), issuer, audience, `exp`/`iat`/`sub`, `role=authenticated`; `alg=none` and HS256 without an explicit legacy secret refused |
| `X-Tenant-ID` | **never** on its own | a selector among the caller's `tenant_memberships`; unknown and forbidden tenants look identical (403). Demo mode trusts it locally only and is refused in production (startup and per request) |
| Chat text | untrusted | length-limited (4000), sent to the model; never interpreted by the application |
| Model output (tool calls, text) | untrusted | schema-validated tools with `extra="forbid"`; no tenant, SQL, URL or file arguments exist; tenant comes from the trusted runtime context; grounding validator checks citations |
| Retrieved policy text | untrusted | returned as data; after retrieval no new commerce tool may run in the turn (`commerce_call_after_retrieval`); actions still need a human |
| Approve / reject | authorised human only | `approver` role from the membership table; tenant-scoped lookup (other tenant → 404); bound to the arguments hash shown on the card |
| Environment | operator-controlled | production refuses unsafe configuration at startup (`app/core/production.py`) |

## Controls

* **Tenant isolation.** Every query filters by the trusted tenant; composite tenant-aware
  foreign keys in the schema; action requests, ledger rows, audit events and checkpoints are
  tenant-scoped (checkpoint thread keys are `sha256(tenant:user-thread)`).
* **No arbitrary capabilities.** Only fixed read tools, one policy search and two narrow,
  schema-only action proposals. No raw SQL, HTTP, file or shell tool reaches the model.
* **Writes need a human.** The model can only PROPOSE; execution happens after an approver's
  decision, in two transactions (claim, then atomic write + status), with an idempotency key.
  Duplicate approvals return the original result; a stale claim fails closed and is never
  re-executed; non-idempotent writes are never silently retried.
* **Errors and logs.** Stable error codes; generic 500 with request id (Phase 12 moved the
  500 path inside the request context so it keeps `X-Request-ID` and security headers).
  Logs never contain prompts, model output, raw payloads, vectors, keys, query strings or
  connection URLs; provider error text is used for classification only.
* **Abuse protection (new).** `POST /api/agent/messages` is rate limited per (user, tenant)
  before any model call (`AGENT_RATE_LIMIT_PER_MINUTE`, default 20; 429 `rate_limited` +
  `Retry-After`). In-process **(V)** — one API instance in the demo.
* **Headers (new).** API: `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
  `Cache-Control: no-store` on `/api/*`. Frontend: CSP (self + configured API + Supabase
  origins, `frame-ancestors 'none'`, `object-src 'none'`), `nosniff`, `DENY`,
  `strict-origin-when-cross-origin`, restrictive `Permissions-Policy`. The e2e suite fails on
  any CSP violation, in dev and production builds.
* **No public API docs in production (new).** `/docs`, `/redoc`, `/openapi.json` are off when
  `APP_ENV=production`.
* **CORS.** Explicit origins only; production requires https origins and no wildcard.
* **CSRF.** Not applicable to the API: authentication is a bearer header, not a cookie.
* **XSS.** React escaping only; no `dangerouslySetInnerHTML`, `eval` or HTML rendering of
  model output; citations are rendered from validated metadata, never from model text.
* **Secrets.** None in the repository (scan below); `.env*` ignored except examples with
  placeholders; settings hold keys as `SecretStr`; `check_env` prints names only.
* **Container.** Non-root user, `$PORT`, no tests or `.env` in the image.

## Public demo (anonymous visitors)

* Identity: a Supabase anonymous session yields a normal signed JWT; only a **verified** claim
  `is_anonymous: true` (a JSON boolean) marks the principal as a public-demo visitor. Headers,
  body fields or strings such as `"true"` never do (tests: `tests/db/test_public_demo.py`).
* Tenant: exactly one configured synthetic tenant (`PUBLIC_DEMO_TENANT_SLUG`), role `member`.
  No `tenant_memberships` row is read or written; selecting any other tenant is 403; a missing
  demo tenant fails closed (503); `PUBLIC_DEMO_ENABLED=false` refuses anonymous users (403).
* Read-only by construction: the demo principal gets a graph profile **without** action tools
  (`commerce-assistant-v3-public-demo`), action resources answer 403 `public_demo_read_only`,
  and `Principal.can_approve` is false even if a role were mis-assigned. A proposal call emitted
  anyway is an unknown tool: nothing is persisted or written.
* Isolation: all visitors share the immutable synthetic business data; conversations never
  mix, because checkpoint threads are keyed by the visitor's own JWT subject.
* Abuse: per-visitor and global demo message budgets (in-process) in addition to Supabase's
  sign-in rate limits.

## Execution trace guarantees

The Agent Execution Trace (`app/agent/trace.py`) is built by the runner from the graph's own
execution records, in executed order (model round → capabilities it requested → next round).
It contains fixed labels plus identifiers, counts, outcomes and measured durations only —
never prompts, messages, model output or reasoning, tool arguments, retrieved text, SQL,
embeddings, tenant ids or secrets (asserted in `tests/db/test_execution_trace.py`, with
mutations that must fail). It is returned with the response and not stored with the
conversation history.

## Live execution trace (streaming)

* Same trust path as the JSON API: auth, tenant selection, rate limit and capability profile
  (public demo = no action tools) are resolved **before** the stream starts; refusals keep
  their JSON status codes. The stream then carries only `RunEvent`s: fixed labels, safe
  details, the trace's closed metadata set, measured durations, and finally the normal
  response body or a safe error code/message/status (tracebacks stay in server logs).
* Never streamed: prompts, system messages, chain-of-thought / model reasoning, raw tool
  messages or arguments, retrieved policy text, SQL, embeddings, tenant ids, JWTs, keys or
  database URLs (`tests/db/test_agent_stream.py::test_stream_contains_only_safe_fields`).
* Reporting cannot change execution: emitter/sink failures are swallowed (a test runs a full
  approve with an exploding sink: one execution, one audit event).
* A browser disconnect does not stop or duplicate anything: the run completes once
  server-side; writes keep their transaction + idempotency guarantees; the UI never resends a
  streamed request (manual "Reload conversation" / manual Retry only). A lost approve stream
  leaves the decision recorded in PostgreSQL; a repeated approve is idempotent.
* The public demo's stream shows only its real read-only path: no action capability, no
  proposal, no approval events.

## Hosted chat providers (Cloudflare Workers AI, Gemini)

* `CLOUDFLARE_API_TOKEN` / `GEMINI_API_KEY` are backend-only `SecretStr` settings, passed
  explicitly to the clients, never read from the browser build (no `NEXT_PUBLIC_*`), never
  logged, never echoed in errors (tests assert the token appears in no exception, result or
  log line). HTTP client libraries (`httpx`, `openai`, `google_genai`) log at WARNING only, so
  request URLs (which contain the Cloudflare account id) are not written to the app log.
* The browser never calls Workers AI; the provider receives only prompts, the synthetic
  conversation and tool schemas — never database credentials, tenant ids or SQL (tool
  schemas are asserted free of `tenant`, `runtime` and SQL terms).
* Provider failures map to the safe error taxonomy; raw provider bodies never reach users.
  Fallback is limited to availability errors at the model-call boundary and cannot replay a
  tool or an action (tests: `tests/db/test_provider_fallback_graph.py`).
* Public demo: in addition to the per-minute limits, each verified anonymous visitor has a
  durable message budget (`PUBLIC_DEMO_MESSAGE_BUDGET`, default 10) keyed by a digest of the
  JWT subject; browser-supplied counters or headers are ignored.

## Verification

| Check | Result |
| --- | --- |
| Backend suite (real PostgreSQL) | see `COMPLETION_STATUS.md` |
| Browser e2e incl. CSP-violation check | 8/8 (dev and production build) |
| Security mutations X1–X14 (full suite per mutation) | see `COMPLETION_STATUS.md` |
| Secret scan (working tree + full git history, key/JWT/private-key/URL-password patterns) | only intentional fake test values |
| `npm audit` (all deps) | 0 vulnerabilities |
| `pip-audit` on the locked runtime dependencies | no known vulnerabilities |

## Residual risks (tracked in `pending-items.md`)

* The rate limit is per process (P16) and there is no global budget cap on the hosted model
  (P17).
* The CSP allows `'unsafe-inline'` scripts (Next.js bootstrap without nonces) (P18).
* Supabase JS keeps the session in `localStorage` (library default); an XSS would expose it —
  mitigated by the CSP and by never rendering HTML (P18).
* Public demo: no CAPTCHA widget yet (P24); anonymous visitors' checkpoint threads are not
  pruned automatically (P25).
* Streaming: an abandoned read-only run still finishes and spends model budget (P28); live
  SSE through the hosted proxies is verified locally only (P26).
* Cloudflare Workers AI tool-calling quality with the chosen model is unmeasured (P30); a new
  anonymous session gets a fresh demo budget (P31).
* `arguments_hash` is optional on approve at the API level (the UI always sends it; arguments
  are immutable after proposal) **(V)**.
