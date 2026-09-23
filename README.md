# CommerceOps AI — Ecommerce AI Operations Agent

CommerceOps AI is a portfolio project that will become an AI operations agent for an
ecommerce business: answering questions and taking actions across customers, orders,
invoices, shipments and company policies, with human approval for sensitive actions.

> **Data notice:** no real customer or company data is used. The project will use
> synthetic ecommerce data only.

## Status: Foundation (Step 1)

What exists today:

- FastAPI backend with environment-based configuration, structured JSON logging,
  CORS, and health endpoints (`/health`, `/health/db`)
- SQLAlchemy 2 + psycopg 3 database layer and an Alembic setup (no migrations yet)
- Local PostgreSQL + pgvector via Docker Compose
- Next.js (App Router, TypeScript, Tailwind) home page showing API and database status
- Backend test suite (pytest)

What does **not** exist yet: domain models, data, business tools, LLM integration,
agents, RAG, embeddings, evaluation or observability. Those are planned, not built.

## Planned capabilities

Roughly in this order: relational ecommerce data model → synthetic seed data → business
tools → LLM provider abstraction (Ollama locally, Gemini for the hosted demo) → LangGraph
agent → RAG over company policies with pgvector → human-in-the-loop approvals → agent
state/memory → evaluation → observability → production deployment.

## Architecture (current)

```
Browser ──▶ Next.js frontend (Vercel) ──fetch──▶ FastAPI backend (Railway / any Docker host)
                                                     │  SQLAlchemy + psycopg 3
                                                     ▼
                                   PostgreSQL + pgvector (local Docker / Supabase)
```

- The backend is stateless; all state lives in PostgreSQL. No Redis, queues or workers.
- Supabase is treated as plain hosted PostgreSQL (no Supabase SDK), so any PostgreSQL
  with the `vector` extension works.
- See [docs/architecture.md](docs/architecture.md) for decisions and layout.

```
backend/   FastAPI app (app/api, core, db, models, schemas, services), tests, Alembic, Dockerfile
frontend/  Next.js app (src/app, src/components, src/lib/api.ts)
docs/      Architecture notes
docker-compose.yml   Local PostgreSQL + pgvector
```

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
| `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL` | backend | Reserved for later steps; unused today |
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
uv sync                     # installs exact versions from uv.lock
uv run uvicorn app.main:app --reload --port 8000
```

- http://localhost:8000/health → `{"status":"ok","service":"commerceops-api"}`
- http://localhost:8000/health/db → `{"status":"ok","database":"reachable"}` (HTTP 503 if not)
- http://localhost:8000/docs → OpenAPI UI

## 4. Frontend

```bash
cd frontend
npm ci
npm run dev                 # http://localhost:3000
```

## 5. Tests and checks

```bash
cd backend
uv run pytest               # unit tests; no database needed
uv run ruff check . && uv run ruff format --check .

# optional integration test against the local DB:
TEST_DATABASE_URL=postgresql://commerceops:<password>@localhost:5432/commerceops uv run pytest -m integration

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
  `DATABASE_URL`, `CORS_ORIGINS` (your Vercel domain). Railway provides `PORT`.
- **Vercel (frontend):** import the repo with root directory `frontend/`, set
  `NEXT_PUBLIC_API_URL` to the Railway URL, then add the Vercel domain to the backend's
  `CORS_ORIGINS`.
