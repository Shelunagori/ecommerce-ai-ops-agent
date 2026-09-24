# syntax=docker/dockerfile:1

# ---- build stage: resolve dependencies from uv.lock -------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime stage ----------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
# Operational entry points used by the pre-deploy step and the runbook (docs/DEPLOYMENT.md):
# check_env, setup_checkpoints, seed_demo, ingest/embed policies, grant_membership.
COPY scripts ./scripts
# Synthetic policy sources for ingestion (no customer data exists in this project).
COPY data/policies ./data/policies

USER app
EXPOSE 8000

# PORT is injected by hosts such as Railway; defaults to 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
