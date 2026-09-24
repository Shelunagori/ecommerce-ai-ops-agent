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
* `arguments_hash` is optional on approve at the API level (the UI always sends it; arguments
  are immutable after proposal) **(V)**.
