# Architecture notes — Step 1 (Foundation)

## Backend layout (`backend/app`)

| Module | Responsibility |
| --- | --- |
| `main.py` | `create_app()` factory: logging, CORS, routers |
| `core/config.py` | `Settings` (pydantic-settings); all config from env / `.env` |
| `core/logging.py` | JSON-lines logging via the standard library |
| `db/url.py` | Normalises `postgres://` / `postgresql://` URLs to `postgresql+psycopg://` |
| `db/session.py` | Lazy engine, session factory, `get_session` dependency, `ping_database()` |
| `models/base.py` | SQLAlchemy `DeclarativeBase` (no domain models yet) |
| `schemas/health.py` | Response models for health endpoints |
| `api/health.py` | `/health`, `/health/db` |
| `services/` | Empty; business logic arrives with the domain model |

## Decisions

- **Stateless API, PostgreSQL for all state.** Runs on cheap container / scale-to-zero
  hosts; no Redis, brokers or workers.
- **Standard PostgreSQL only.** psycopg 3 via SQLAlchemy; no Supabase SDK. Local image is
  `pgvector/pgvector:pg17`, but nothing depends on 17-specific features — only standard
  SQL and the `vector` extension.
- **Engine created lazily.** The app starts without a database; `/health` stays green
  while `/health/db` reports 503 (`unreachable` or `not_configured`).
- **Safe errors.** DB failures are logged by exception type only (driver messages can
  contain host/user); the URL normaliser never echoes the URL.
- **Testable dependency.** `/health/db` gets its check through `Depends(get_db_check)`,
  so tests swap it without a database. A real-DB test runs only with `TEST_DATABASE_URL`.
- **CORS from env.** Explicit origin list; if `*` is configured, credentials are disabled.
- **Reproducible installs.** `uv.lock` (backend) and `package-lock.json` (frontend).
- **Frontend API access in one place.** `src/lib/api.ts` reads `NEXT_PUBLIC_API_URL`;
  falls back to `http://localhost:8000` only under `next dev`. Status is fetched from the
  browser, so it exercises CORS the same way production will.
- **Migrations.** Alembic reads `DATABASE_URL` from settings; the first migration arrives
  with the domain model (Step 2). Enabling `vector` will be a migration (Step 8).
