# CommerceOps AI — Ecommerce AI Operations Agent

CommerceOps AI is a portfolio project that will become an AI operations agent for an
ecommerce business: answering questions and taking actions across customers, orders,
invoices, shipments and company policies, with human approval for sensitive actions.

> **Data notice:** no real customer or company data is used. The project will use
> synthetic ecommerce data only.

## Status: Step 9 — end-to-end RAG in the LangGraph assistant

What exists today:

- FastAPI backend with environment-based configuration, structured JSON logging
  (with request id and tenant id), CORS, and health endpoints (`/health`, `/health/db`)
- **Ecommerce domain model** in PostgreSQL: tenants, customers, products, orders,
  order items, invoices, shipments — with database-enforced tenant integrity
- Alembic migrations `0001` (ecommerce schema), `0002` (knowledge documents/chunks) and
  `0003` (pgvector chunk embeddings; enables the `vector` extension)
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
  by a structured intent-classification test vehicle and by the commerce assistant.
- A **model-driven commerce assistant** (`app/agent/assistant`): the model receives the 12
  tool schemas (`bind_tools`), requests tool calls, the host executes them through the
  Step 3 tools with the trusted tenant context, and the model answers from the results.
  The tool-calling loop is explicit and bounded, single-turn, read-only; it is kept as the
  reference implementation.
- A **LangGraph** version of the same assistant (`app/agent/graph`): a `StateGraph` with a
  model node and a tool node, typed state, trusted tenant context as LangGraph runtime
  context, the same limits and result contract, parity-tested against the Step 5 loop.
  Optional in-memory checkpointing with tenant-scoped thread IDs (ephemeral: lost when the
  process exits).
- A **knowledge / RAG foundation** (`app/knowledge`): a synthetic, versioned policy corpus
  for both tenants (`backend/data/policies`), a validated loader, a deterministic
  section-aware chunker (`policy-section-v1`), atomic idempotent ingestion into
  `knowledge_documents` / `knowledge_chunks`, effective-date version selection,
  tenant-relative citations and a deterministic **lexical** baseline retriever
  (PostgreSQL full-text search), with a 20-case retrieval evaluation set. It retrieves
  chunks only: nothing is generated and the assistant does not use it yet.
- **Embeddings and semantic retrieval** (`app/knowledge/embeddings`, `app/knowledge/semantic.py`):
  local Ollama `nomic-embed-text-v2-moe` (768 dimensions) behind a provider-neutral
  contract, vectors stored in PostgreSQL with pgvector in a separate
  `knowledge_chunk_embeddings` table per concrete embedding profile (including the resolved
  model digest), and an exact cosine-similarity retriever (`semantic-pgvector-v1`) with the
  same tenant, effective-date, limit and citation rules as the lexical baseline. Both
  retrievers are scored on the same 20 evaluation cases. On the fixed 20-case synthetic
  evaluation corpus, semantic retrieval achieved exact-chunk hit@1 of 1.00 versus 0.40 for
  lexical retrieval (12 semantic wins, 8 ties, 0 losses per case; document hit@1 is 1.00
  for both). This is a small synthetic set, not a general accuracy claim — see
  [architecture](docs/architecture.md#embeddings-and-semantic-retrieval-step-8).
  Retrieval only: no hybrid ranking, no ANN index, no generated answers.
- **End-to-end RAG** (`app/agent/rag`, LangGraph RETRIEVE node): the graph assistant
  (prompt `commerce-assistant-v2`) answers policy questions by calling
  `search_policy_knowledge`; semantic retrieval runs with the trusted tenant and the
  requested effective date, and the answer must cite `policy://…` chunks retrieved in the
  **current** user turn (validated deterministically; earlier-turn citations are stale).
  Mixed questions use commerce tools first, then retrieval, in separate rounds. Citations
  show which retrieved chunks an answer relies on; they do not prove every sentence is
  correct. The Step-5 loop is unchanged.
- Next.js page showing API/database status and per-tenant demo data counts
- Backend tests, including real-PostgreSQL tests for constraints and cross-tenant isolation

What does **not** exist yet: hybrid lexical+semantic ranking, rerankers, ANN (HNSW /
IVFFlat) indexes, business actions/writes, human approval, durable checkpoints or long-term
memory, frontend chat, full evaluation or observability. Those are planned, not built.

## Planned capabilities

Done: relational ecommerce data model, synthetic seed data, read-only business tools, LLM
provider layer (Ollama / Gemini), explicit model-driven tool calling, LangGraph
orchestration, knowledge/RAG foundation with a lexical baseline, embeddings + pgvector
semantic retrieval, end-to-end RAG answers with validated citations. Next, roughly in
order: human-in-the-loop approvals → durable agent state/memory → evaluation → observability →
production deployment.

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
| Policy knowledge (refunds, returns, shipping, …) | `knowledge_documents` / `knowledge_chunks`, lexical retrieval (Step 7) |
| Policy embeddings | `knowledge_chunk_embeddings` (pgvector), exact cosine retrieval (Step 8) |
| RAG answers / hybrid ranking | **Not implemented yet** |

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
text. Policies are the opposite: prose that answers "what is the rule?", so they go
through knowledge retrieval:

```
"What is the total of ORD-1001?"        → commerce tool → relational query (exact fact)
"What is the cancellation policy?"      → knowledge retrieval → cited policy chunks
```

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
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | backend | Local Ollama server and model (default `qwen3:4b-instruct`, the non-thinking Qwen3-4B-Instruct-2507; any other Ollama model, e.g. `llama3.2:3b`, is selectable) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | backend | Gemini key (required only for `gemini`) and model (default `gemini-3.8-flash`) |
| `ASSISTANT_MAX_MODEL_ROUNDS`, `ASSISTANT_MAX_TOOL_CALLS`, `ASSISTANT_MAX_TOOL_CALLS_PER_TURN` | backend | Assistant loop bounds (defaults 5 / 8 / 4; hard maxima 10 / 20 / 8) |
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

### Commerce assistant (model + tools, single turn)

```bash
uv run python -m scripts.run_assistant --tenant $NORTHSTAR --provider ollama \
  --text "Show me order ORD-1001"
uv run python -m scripts.run_assistant --tenant $BLUEPEAK --provider gemini \
  --text "What is the latest unpaid invoice for CUS-1002?"
```

stdout is JSON: answer, provider/model, prompt version, model-call count, tool-call summary
(tool, business arguments, outcome, duration) and total duration. `--tenant` is trusted
runtime context and is validated before any model is built; it never reaches the prompt.
Ask about policies (refunds, compensation) and it will say that knowledge is not available
yet — there is no RAG.

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
