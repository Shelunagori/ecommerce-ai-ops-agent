# Development guide

Local setup, command reference and test commands. For the overview see the
[README](../README.md); for hosting see [DEPLOYMENT.md](DEPLOYMENT.md).

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (uv installs Python 3.12 from `.python-version` if needed)
- Node.js 22.22.2+ (or 24.15+) and npm — `nvm use` reads `frontend/.nvmrc`; Node 20 is not supported (jsdom/vitest need 22+)
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
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | backend | Local Ollama server and model (default `qwen3:4b-instruct`, the non-thinking Qwen3-4B-Instruct-2507; any other Ollama model, e.g. `llama3.2:3b`, is selectable) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | backend | Gemini key (required only for `gemini`) and model (default `gemini-3.8-flash`) |
| `ASSISTANT_MAX_MODEL_ROUNDS`, `ASSISTANT_MAX_TOOL_CALLS`, `ASSISTANT_MAX_TOOL_CALLS_PER_TURN` | backend | Assistant loop bounds (defaults 5 / 8 / 4; hard maxima 10 / 20 / 8) |
| `EMBEDDING_PROVIDER`, `OLLAMA_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | backend | Policy embeddings (`ollama` local default, `gemini` hosted) |
| `ACTION_APPROVAL_TTL_SECONDS`, `ACTION_EXECUTION_CLAIM_TIMEOUT_SECONDS`, `STORE_CREDIT_MAX_AMOUNT` | backend | Approval-gated actions (defaults 900 s / 120 s / 100.00) |
| `AUTH_MODE`, `SUPABASE_URL`, `SUPABASE_JWT_AUDIENCE`, `SUPABASE_JWT_SECRET`, `DEMO_USER_SUBJECT` | backend | `demo` (local header) or `supabase` (JWT + memberships) |
| `AGENT_RATE_LIMIT_PER_MINUTE` | backend | Chat messages per user and tenant per minute (default 20; 0 disables) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | backend | External tracing, opt-in; keep `false` |
| `NEXT_PUBLIC_API_URL` | frontend | Backend base URL (inlined at build time) |
| `NEXT_PUBLIC_AUTH_MODE`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | frontend | `demo` or `supabase` sign-in (public values only) |

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
uv run python -m scripts.ingest_policies    # synthetic policy corpus
uv run python -m scripts.embed_policies     # needs `ollama pull nomic-embed-text-v2-moe`
uv run python -m scripts.setup_checkpoints  # durable LangGraph checkpoint tables (idempotent)
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

Read-only endpoints (demo mode: `X-Tenant-ID`; supabase mode: bearer token + membership): `/api/customers[?search=]`,
`/api/customers/{code}`, `/api/customers/{code}/orders`,
`/api/customers/{code}/invoices/unpaid`, `/api/orders?status=`, `/api/orders/{number}`,
`/api/invoices/{number}`, `/api/shipments[?status=]`, `/api/shipments/{number}`,
`/api/products[?search=]`, `/api/products/{sku}`, `/api/demo/summary`.
Errors use one envelope: `{"error": {"code", "message"}, "request_id"}`.

- http://localhost:8000/health → `{"status":"ok","service":"commerceops-api"}`
- http://localhost:8000/health/db → `{"status":"ok","database":"reachable"}` (HTTP 503 if not)
- http://localhost:8000/health/ready → readiness (config, database, migrations, checkpoints)
- http://localhost:8000/docs → OpenAPI UI (not published when `APP_ENV=production`)

### Agent tools (direct invocation, no model)

```bash
uv run python -m scripts.run_tool --list      # tools + model-visible parameters
uv run python -m scripts.run_tool --tenant $NORTHSTAR --tool get_order \
  --args '{"order_number": "ORD-1010"}'
uv run python -m scripts.run_tool --tenant $BLUEPEAK --tool list_delayed_shipments
```

The tenant comes from `--tenant` (trusted runtime context), never from `--args`.
Developer tooling only — not an HTTP endpoint.
External tracing (LangSmith) is opt-in and off by default (`LANGSMITH_TRACING=false`).

### LLM provider smoke test (structured output, no tools)

Local, free — [Ollama](https://ollama.com):

```bash
ollama pull qwen3:4b-instruct   # ~2.5 GB, once (or: ollama pull llama3.2:3b + OLLAMA_MODEL=llama3.2:3b)
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

### Step-5 reference assistant (model + tools, single turn)

```bash
uv run python -m scripts.run_assistant --tenant $NORTHSTAR --provider ollama \
  --text "Show me order ORD-1001"
uv run python -m scripts.run_assistant --tenant $BLUEPEAK --provider gemini \
  --text "What is the latest unpaid invoice for CUS-1002?"
```

stdout is JSON: answer, provider/model, prompt version, model-call count, tool-call summary
(tool, business arguments, outcome, duration) and total duration. `--tenant` is trusted
runtime context and is validated before any model is built; it never reaches the prompt.
This is the unchanged Step-5 reference loop (no RAG, no actions); the product path is the
LangGraph assistant behind the API.

### Policy knowledge (ingest and search; no LLM)

```bash
uv run python -m scripts.seed_demo          # tenants must exist first
uv run python -m scripts.ingest_policies    # idempotent; refuses APP_ENV=production
uv run python -m scripts.search_policies --tenant $NORTHSTAR \
  --query "compensation for delayed shipment" --as-of 2026-09-01
```

Ingestion prints a JSON summary (`inserted` / `unchanged` / `retired` / conflicts) and is
atomic: an edited, already-ingested version or a changed chunking configuration aborts the
run without writing. Search prints ranked chunks with score, citation
(`policy://<document_key>/v<version>#chunk-<n>`), version and effective dates. `--as-of`
takes a date or a timezone-aware datetime (naive datetimes are rejected).

### Policy embeddings and semantic search (local Ollama; no LLM answer)

```bash
ollama pull nomic-embed-text-v2-moe         # local embedding model (768 dimensions)
uv run python -m scripts.ingest_policies
uv run python -m scripts.embed_policies     # idempotent; refuses APP_ENV=production
uv run python -m scripts.search_policies --retriever semantic --tenant $BLUEPEAK \
  --query "express delivery compensation"
uv run python -m scripts.eval_retrieval --retriever semantic   # 20 cases, vs lexical
```

`embed_policies` resolves the model digest once, embeds only chunks missing for that
concrete profile and prints a JSON summary. Re-pulling the model under the same tag gives a
new digest, i.e. a new profile: run `embed_policies` again (old vectors stay untouched).
Semantic search refuses to run (`embedding_profile_not_materialized`) until the current
profile has been materialized; it never falls back to vectors from another model build.

### LangGraph assistant

```bash
uv run python -m scripts.run_graph_assistant --tenant $NORTHSTAR --provider ollama \
  --text "Show me order ORD-1001"
# Same-process continuation on an in-memory thread (gone when the process exits):
uv run python -m scripts.run_graph_assistant --tenant $NORTHSTAR --thread-id demo \
  --text "Show me order ORD-1001" --text "Is it paid?"
```

Same JSON result as `run_assistant` plus `retrievals` (no query text) and `citations`
(a list when `--text` is repeated). `--thread-id` enables LangGraph's `InMemorySaver` for
this process only; nothing is written to disk or the database.

Policy questions (needs `ingest_policies` + `embed_policies` first):

```bash
uv run python -m scripts.run_graph_assistant --tenant $BLUEPEAK --provider ollama \
  --text "What compensation applies to a shipment delayed by 10 days?"
# Live RAG measurement on the 12 RAG/agent cases (structured outcomes, no prose):
uv run python -m scripts.eval_rag --provider ollama --write data/eval/rag_live_measurement_v1.json
```

### Agent API: chat and approvals (what the frontend uses)

```bash
curl -s localhost:8000/api/me                                   # demo mode: both tenants, role approver
curl -s -X POST localhost:8000/api/agent/messages -H "X-Tenant-ID: $NORTHSTAR" \
  -H 'content-type: application/json' -d '{"text":"Please cancel ORD-1004","thread_id":"demo-1"}'
# -> "action": {"id": ..., "status": "pending_approval", "arguments_hash": ...}
curl -s -X POST localhost:8000/api/agent/actions/<id>/approve -H "X-Tenant-ID: $NORTHSTAR" \
  -H 'content-type: application/json' -d '{"arguments_hash":"<hash from the card>"}'
curl -s localhost:8000/api/agent/threads/demo-1/messages -H "X-Tenant-ID: $NORTHSTAR"   # sanitised history
curl -s "localhost:8000/api/agent/actions?status=succeeded" -H "X-Tenant-ID: $NORTHSTAR"
```

With `AUTH_MODE=supabase`, replace the header with `Authorization: Bearer <Supabase access
token>` (plus `X-Tenant-ID` to pick one of your memberships) and grant access first:
`uv run python -m scripts.grant_membership --tenant northstar-commerce --subject <user-uuid> --role approver`.

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
# Live tool-calling loops (Step 5 loop and Step 6 graph) need TEST_DATABASE_URL as well:
RUN_OLLAMA_INTEGRATION=1 TEST_DATABASE_URL=... uv run pytest tests/db/test_assistant_live.py tests/db/test_graph_live.py
# Live embeddings (needs `ollama pull nomic-embed-text-v2-moe`):
RUN_OLLAMA_INTEGRATION=1 TEST_DATABASE_URL=... uv run pytest tests/db/test_embeddings_live.py
# Live RAG (chat model = LIVE_RAG_MODEL if set, else OLLAMA_MODEL; plus the embedding model):
RUN_OLLAMA_INTEGRATION=1 TEST_DATABASE_URL=... uv run pytest tests/db/test_rag_live.py

cd ../frontend
npm run lint && npm run typecheck && npm test && npm run build
# Browser end-to-end (real API + real PostgreSQL, deterministic chat model; resets the _test DB):
TEST_DATABASE_URL=postgresql://commerceops:<password>@localhost:5432/commerceops_test npm run test:e2e
# once per machine: npx playwright install chromium
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
With `APP_ENV=production` it refuses to start on an unsafe configuration; see
[DEPLOYMENT.md](DEPLOYMENT.md).
