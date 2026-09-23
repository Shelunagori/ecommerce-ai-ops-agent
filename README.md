# CommerceOps AI — Ecommerce AI Operations Agent

CommerceOps AI is a portfolio project that will become an AI operations agent for an
ecommerce business: answering questions and taking actions across customers, orders,
invoices, shipments and company policies, with human approval for sensitive actions.

> **Data notice:** no real customer or company data is used. The project will use
> synthetic ecommerce data only.

## Status: Step 4 — LLM provider layer

What exists today:

- FastAPI backend with environment-based configuration, structured JSON logging
  (with request id and tenant id), CORS, and health endpoints (`/health`, `/health/db`)
- **Ecommerce domain model** in PostgreSQL: tenants, customers, products, orders,
  order items, invoices, shipments — with database-enforced tenant integrity
- Alembic migration `0001` for the whole schema (no extensions enabled)
- A tenant-scoped, read-only **query layer** (`app/services`) that future agent tools will
  call directly, plus a small set of read-only `/api/...` endpoints over it
- An idempotent **synthetic** seed script with two demo tenants
- **12 read-only agent tools** (`app/agent/tools`, LangChain `@tool` + `ToolRuntime`) over
  the query layer, with typed bounded inputs, a stable JSON envelope and tenant context
  injected at runtime (never a model argument) — plus a developer CLI to run them. No model
  calls them yet.
- A provider-neutral **LLM layer** (`app/agent/llm`): Ollama for local development, Gemini
  for the hosted demo, native structured output validated against a Pydantic schema,
  explicit timeouts, a conservative retry policy and typed safe errors. It is exercised
  only by a structured intent-classification test vehicle; **no tools are bound to the
  model and there is no agent yet.**
- Next.js page showing API/database status and per-tenant demo data counts
- Backend tests, including real-PostgreSQL tests for constraints and cross-tenant isolation

What does **not** exist yet: business actions/writes, tool-calling by a model, agents or
LangGraph workflows, RAG,
embeddings, vector search, approvals, evaluation or observability. Those are planned,
not built.

## Planned capabilities

Done: relational ecommerce data model, synthetic seed data, read-only business tools, LLM
provider layer (Ollama / Gemini). Next, roughly in order: LangGraph agent → RAG over company policies with pgvector → human-in-the-loop approvals → agent
state/memory → evaluation → observability → production deployment.

## Architecture (current)

```
Browser ──▶ Next.js frontend (Vercel) ──fetch──▶ FastAPI backend (Railway / any Docker host)
                                                   │  api/  → read-only routes, X-Tenant-ID demo header
                                                   │  services/ → tenant-scoped query classes
                                                   │  SQLAlchemy 2 + psycopg 3
                                                   ▼
                                 PostgreSQL (+ pgvector image; extension not enabled yet)
```

| Concern | Where it lives today |
| --- | --- |
| Structured business data (orders, invoices, shipments, …) | PostgreSQL tables, queried deterministically |
| AI / RAG / embeddings | **Not implemented yet** |

- The backend is stateless; all state lives in PostgreSQL. No Redis, queues or workers.
- Supabase is treated as plain hosted PostgreSQL (no Supabase SDK).
- Details and decisions: [docs/architecture.md](docs/architecture.md); open follow-ups: [docs/pending-items.md](docs/pending-items.md).

```
backend/   app/{api,core,db,models,schemas,services}, alembic/, scripts/seed_demo.py, tests/
frontend/  Next.js app (src/app, src/components, src/lib/api.ts, src/lib/demo.ts)
docs/      Architecture notes
docker-compose.yml   Local PostgreSQL + pgvector image
```

### Why structured facts use database queries, not RAG

Order status, invoice amounts, due dates and shipment state are authoritative relational
facts. They change, must be exact, and must be scoped to one tenant. They are therefore
read from PostgreSQL through deterministic, tenant-scoped queries — not retrieved
probabilistically from embeddings, which can return stale, approximate or wrong-tenant
text. RAG (later) is intended for unstructured knowledge such as company policies.

### Multi-tenancy

Every business row carries `tenant_id`, and child rows reference their parents through
composite `(tenant_id, id)` foreign keys, so PostgreSQL itself rejects e.g. a Tenant A
order pointing at a Tenant B customer. Every query method is bound to a `TenantContext`.

> **The `X-Tenant-ID` header is NOT authentication.** It is a temporary demo mechanism
> for propagating tenant context: any caller can send any tenant id. It will be replaced
> by authenticated tenant context; only `app/api/deps.py` has to change for that.

### Data

All data in this repository is **synthetic** (seeded by `backend/scripts/seed_demo.py`).
Names, emails (`example.com` / `example.org`), SKUs and tracking numbers are invented.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (uv installs Python 3.12 from `.python-version` if needed)
- Node.js 20.9+ and npm
- Docker (for local PostgreSQL)

## 1. Environment variables

Three example files, all placeholders — copy them and never commit the copies:

```bash
cp .env.example .env                        # docker compose: POSTGRES_* for the local DB
cp backend/.env.example backend/.env        # backend settings
cp frontend/.env.example frontend/.env.local
```

Make the password in `backend/.env`'s `DATABASE_URL` match `POSTGRES_PASSWORD` in `.env`.

| Variable | Where | Purpose |
| --- | --- | --- |
| `APP_ENV` | backend | `development` / `test` / `staging` / `production` |
| `APP_NAME` | backend | Service name |
| `DEBUG` | backend | Verbose logging when `true` |
| `DATABASE_URL` | backend | PostgreSQL URL. `postgres://`, `postgresql://` are normalised to `postgresql+psycopg://` |
| `CORS_ORIGINS` | backend | Comma-separated allowed origins, e.g. `http://localhost:3000,https://your-app.vercel.app` |
| `LLM_PROVIDER` | backend | `ollama` (local, default) or `gemini` (hosted demo) |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | backend | Per-request timeout, 1–300 s (default 60); retries for transient failures only (default 1, max 2) |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | backend | Local Ollama server and model (default `llama3.2:3b`) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | backend | Gemini key (required only for `gemini`) and model (default `gemini-3.8-flash`) |
| `LANGSMITH_TRACING` | process env | External tracing, opt-in; keep `false` |
| `NEXT_PUBLIC_API_URL` | frontend | Backend base URL (inlined at build time) |

## 2. Database (Docker)

```bash
docker compose up -d db
docker compose ps          # wait for "healthy"
```

Data persists in the `commerceops-pgdata` volume (`docker compose down -v` deletes it).
Port 5432 is bound to `127.0.0.1` only.

## 3. Backend

```bash
cd backend
uv sync                              # installs exact versions from uv.lock
uv run alembic upgrade head          # create/upgrade the schema
uv run python -m scripts.seed_demo   # idempotent synthetic demo data (refuses APP_ENV=production)
uv run uvicorn app.main:app --reload --port 8000
```

Try it (demo tenant ids are deterministic and printed by the seed script):

```bash
NORTHSTAR=17243d88-ed66-5445-955b-7d7572094122
BLUEPEAK=11a6d918-f89f-59b5-ab1e-45a109978c8a
curl -s localhost:8000/api/orders/ORD-1001 -H "X-Tenant-ID: $NORTHSTAR"
curl -s localhost:8000/api/orders/ORD-1001 -H "X-Tenant-ID: $BLUEPEAK"   # different order
curl -s "localhost:8000/api/shipments?status=delayed" -H "X-Tenant-ID: $NORTHSTAR"
```

Read-only endpoints (all require `X-Tenant-ID`): `/api/customers[?search=]`,
`/api/customers/{code}`, `/api/customers/{code}/orders`,
`/api/customers/{code}/invoices/unpaid`, `/api/orders?status=`, `/api/orders/{number}`,
`/api/invoices/{number}`, `/api/shipments[?status=]`, `/api/shipments/{number}`,
`/api/products[?search=]`, `/api/products/{sku}`, `/api/demo/summary`.
Errors use one envelope: `{"error": {"code", "message"}, "request_id"}`.

- http://localhost:8000/health → `{"status":"ok","service":"commerceops-api"}`
- http://localhost:8000/health/db → `{"status":"ok","database":"reachable"}` (HTTP 503 if not)
- http://localhost:8000/docs → OpenAPI UI

### Agent tools (no model yet)

```bash
uv run python -m scripts.run_tool --list      # tools + model-visible parameters
uv run python -m scripts.run_tool --tenant $NORTHSTAR --tool get_order \
  --args '{"order_number": "ORD-1010"}'
uv run python -m scripts.run_tool --tenant $BLUEPEAK --tool list_delayed_shipments
```

The tenant comes from `--tenant` (trusted runtime context), never from `--args`.
Developer tooling only — not an HTTP endpoint.
External tracing (LangSmith) is opt-in and off by default (`LANGSMITH_TRACING=false`).

### LLM provider (no tools, no agent yet)

Local, free — [Ollama](https://ollama.com):

```bash
ollama pull llama3.2:3b     # ~2 GB, once
ollama serve                # if the Ollama app is not already running
uv run python -m scripts.run_llm --provider ollama --text "Show me order ORD-1001"
```

Hosted demo — Gemini (set `GEMINI_API_KEY` in `backend/.env`, never commit it):

```bash
uv run python -m scripts.run_llm --provider gemini --text "Where is shipment SHP-1003?"
```

> **Gemini: synthetic/demo data only.** The free Gemini API tier may use submitted content
> to improve Google's products. Never send real customer, employer, client (e.g.
> Brandhub) or other confidential data — only the synthetic CommerceOps demo data.

stdout is the validated result JSON; one safe log line goes to stderr. Keys, prompts and
raw model responses are never printed or logged.

## 4. Frontend

```bash
cd frontend
npm ci
npm run dev                 # http://localhost:3000
```

## 5. Tests and checks

```bash
cd backend
uv run ruff check . && uv run ruff format --check .
uv run pytest               # without TEST_DATABASE_URL: unit tests only, DB tests are skipped

# Full suite against real PostgreSQL. Use a DEDICATED database whose name ends in _test:
# the suite runs `alembic downgrade base`, so it refuses any other database name.
docker compose exec db createdb -U commerceops commerceops_test     # once
TEST_DATABASE_URL=postgresql://commerceops:<password>@localhost:5432/commerceops_test uv run pytest

# Optional live provider smoke tests (skipped by default; the normal suite never calls a model).
# They check provider integration and the structured-output contract, not model quality:
RUN_OLLAMA_INTEGRATION=1 uv run pytest tests/integration -m llm_integration
RUN_GEMINI_INTEGRATION=1 uv run pytest tests/integration -m llm_integration   # uses quota

cd ../frontend
npm run lint && npm run typecheck && npm run build
```

## 6. Backend Docker image

```bash
cd backend
docker build -t commerceops-api .
docker run --rm -p 8000:8000 --env-file .env \
  -e DATABASE_URL=postgresql://commerceops:<password>@host.docker.internal:5432/commerceops \
  commerceops-api
```

The image runs as a non-root user, contains no secrets, and listens on `$PORT` (default 8000).

## Deployment targets (later steps)

These are the intended targets; nothing is deployed yet.

- **Supabase (PostgreSQL + pgvector):** create a project, enable the `vector` extension,
  and use its connection string as `DATABASE_URL`. Prefer the direct or *session pooler*
  (port 5432) connection; the *transaction pooler* (6543) needs prepared statements
  disabled, which will be handled when it is needed.
- **Railway (backend):** deploy `backend/` using its Dockerfile; set `APP_ENV=production`,
  `DATABASE_URL`, `CORS_ORIGINS` (your Vercel domain), `LLM_PROVIDER=gemini`,
  `GEMINI_API_KEY` (as a Railway secret) and `LANGSMITH_TRACING=false`. Railway provides
  `PORT`. The hosted backend never needs Ollama; no model files are baked into the image.
- **Vercel (frontend):** import the repo with root directory `frontend/`, set
  `NEXT_PUBLIC_API_URL` to the Railway URL, then add the Vercel domain to the backend's
  `CORS_ORIGINS`.
