# Architecture notes

Current step: **Step 2 — structured ecommerce data.** No AI, RAG, embeddings or agents yet.

| Concern | Where it lives |
| --- | --- |
| Structured business data | PostgreSQL (7 tables, Alembic migration `0001`) |
| Access to that data | `app/services` tenant-scoped query classes (read-only) |
| AI / RAG / embeddings | Not implemented yet (pgvector image present, extension not enabled) |

## Backend layout (`backend/`)

| Module | Responsibility |
| --- | --- |
| `app/main.py` | `create_app()`: logging, CORS, request-context middleware, error handlers, routers |
| `app/core/config.py` | `Settings` (pydantic-settings) |
| `app/core/logging.py` | JSON-lines logging; adds `request_id` / `tenant_id` to every record |
| `app/core/request_context.py` | Context vars for request id and (claimed) tenant id |
| `app/core/tenant.py` | `TenantContext` — the only tenant input the services accept |
| `app/core/errors.py` | `AppError` hierarchy with safe client messages |
| `app/db/session.py` | Lazy engine; `read_only_session()`, `unit_of_work()`, `get_read_session` |
| `app/models/` | SQLAlchemy models, enums, naming convention |
| `app/schemas/commerce.py` | Frozen Pydantic read models (what services and API return) |
| `app/services/` | `CustomerQueries`, `ProductQueries`, `OrderQueries`, `InvoiceQueries`, `ShipmentQueries`, `SummaryQueries`, facade `CommerceQueries`, `resolve_tenant_context` |
| `app/api/deps.py` | `X-Tenant-ID` → `TenantContext` (demo mechanism), clock, `Queries` dependency |
| `app/api/errors.py` | Uniform error envelope, 422/404/500 handlers |
| `app/api/middleware.py` | Request id, `X-Request-ID` response header, one access-log line per request |
| `app/api/routes/commerce.py` | Read-only `/api/...` endpoints |
| `scripts/seed_demo.py` | Idempotent synthetic seed |

## Domain model

```
tenants ─┬─< customers ─< orders ─┬─< order_items >─ products >─┐
         │                        ├─< invoices                  │
         │                        └─< shipments                 │
         └──────────────────────────────────────────────────────┘
```

- UUID primary keys (`gen_random_uuid()` server default, core PostgreSQL ≥ 13), `TIMESTAMPTZ`
  timestamps, `created_at` / `updated_at` on every table.
- Money is `NUMERIC(12,2)` / `Decimal`, serialised to JSON as strings.
- Statuses are `VARCHAR` + named `CHECK` constraints backed by Python `StrEnum`s (no native
  PG enums, so adding a status is a simple constraint swap in a migration).
- Human references (`customer_code`, `sku`, `order_number`, `invoice_number`,
  `shipment_number`) are unique **per tenant**; `tenants.slug` is globally unique.
- Customer email is indexed but deliberately not unique (shared/reused addresses exist).

### Tenant integrity (database-enforced)

1. Parents expose `UNIQUE (tenant_id, id)` (customers, products, orders).
2. Children reference them with composite FKs on `(tenant_id, <parent>_id)`:
   `orders → customers`, `order_items → orders` and `→ products`,
   `invoices → orders`, `shipments → orders`. `customers` / `products` reference `tenants`.
3. Because the child's own `tenant_id` is part of the FK, PostgreSQL rejects any row whose
   parent belongs to a different tenant — including re-pointing an existing row's
   `tenant_id`. This does not depend on application code being correct.
4. The `(tenant_id, id)` unique indexes double as the tenant-leading index for lookups, so
   no separate `tenant_id` indexes are needed.

### Lifecycle vs derived state

- Invoice status is lifecycle only: `pending | paid | cancelled`. **Overdue is derived**
  (`status = 'pending' AND due_at < now`) at read time; nothing needs a background job to
  keep it current. `is_overdue` is exposed on the invoice read model. "Unpaid" = pending.
- Contradictory states prevented by CHECKs (not a full state machine):
  paid ⇔ `paid_at` present; `due_at >= issued_at`; non-draft orders have `placed_at`;
  delivered shipments have `delivered_at`; only delivered/returned shipments may carry
  `delivered_at`; delivery not before shipping; `line_total = quantity × unit_price`;
  quantity > 0; money ≥ 0; ISO-4217-shaped currency.
- Shipment `delayed` is a **stored operational state** (carrier- or operations-reported),
  not a derived one. `delay_reason` is nullable even for `delayed` shipments (a delay can be
  known before its cause). A possible later *derived* flag such as
  `is_late = delivered_at IS NULL AND expected_delivery_at < now` is a different concept:
  it can be true without any carrier report, and a `delayed` shipment need not be late yet.
- Order status and shipment status are **not** coupled by database constraints. Consistency
  between them is an application/workflow rule, especially since an order may have several
  shipments.
- Order items: the line's own UUID identifies it. There is no (order, product) uniqueness —
  the same product may appear on several lines (price, discount, customisation,
  fulfilment, tax). Demo line ids derive from (tenant, order, line number), never the SKU;
  demo order `ORD-1010` (Northstar) lists `SKU-1005` on two lines.
- Order `total_amount` = Σ line totals is maintained by the writing code (today: the seed)
  and covered by tests, not by a trigger. Production systems may choose a stronger
  strategy (trigger, or computing totals in one write service) depending on write workflows.
- `updated_at` is maintained by SQLAlchemy on application-issued updates (no DB trigger).

### Indexes

Beyond the unique constraints: `customers (tenant_id, name)`, `(tenant_id, email)`;
`products (tenant_id, name)`; `orders (tenant_id, customer_id, placed_at)` (history, latest
order, and the composite FK), `(tenant_id, status)`, `(tenant_id, placed_at)`;
`order_items (tenant_id, order_id)` (order lines and the composite FK), `(tenant_id, product_id)`; `invoices (tenant_id, status, due_at)` (unpaid /
overdue), `(tenant_id, due_at)`, `(tenant_id, order_id)`; `shipments (tenant_id, status)`,
`(tenant_id, order_id)`. Name search uses `ILIKE` (user wildcards escaped); no `pg_trgm`
until a query requirement justifies it.

## Why structured facts use deterministic database tools, not RAG

Order status, invoice amount and shipment state are authoritative relational facts. They
must be exact, current and tenant-scoped, so they are queried from PostgreSQL by explicit
tenant-filtered queries. Retrieving them probabilistically from embeddings could return
stale, approximate or cross-tenant text. RAG (a later step) is reserved for unstructured
knowledge such as company policies; agent tools will call `app/services` for facts.

## Tenant context (demo) — NOT authentication

- `X-Tenant-ID: <uuid>` is read by exactly one dependency, `get_tenant_context`
  (`app/api/deps.py`). Missing/blank → `400 tenant_context_missing`; not a UUID →
  `400 tenant_context_invalid`; unknown tenant → `404 tenant_not_found` (one PK lookup in
  `resolve_tenant_context`).
- Anyone can send any tenant id: this is a demo mechanism for propagating tenant context,
  not a security boundary. Authenticated identity will later produce the same
  `TenantContext` by replacing that dependency; services do not change.
- A reference that exists only in another tenant returns the same 404 as one that exists
  nowhere.

## Service layer and transactions

- Query classes take `(session, TenantContext, clock)` in their constructor; the tenant is
  fixed for the object's lifetime, so no public method can run without it. Joins are on
  `(tenant_id, id)` and every statement filters by the tenant.
- They return frozen Pydantic read models (no ORM objects escape, no lazy loading after the
  session closes). Internal UUIDs and tenant ids are not exposed.
- **Transaction boundary: the caller owns it.** Query classes never `commit()` / `flush()`.
  - Reads: `read_only_session()` issues `SET TRANSACTION READ ONLY` and always rolls back —
    an accidental write fails in PostgreSQL. The API uses it per request.
  - Writes (today only the seed): `unit_of_work()` commits once on success, rolls back on
    error.
  - Future agent tools compose several queries inside one session/transaction the same way.
- Time-dependent derivations take an injectable clock (tests use a fixed one).

## Errors and logging

- Error envelope: `{"error": {"code", "message"}, "request_id"}`. Messages never include
  SQL, connection details, stack traces or other tenants' data; 422s list field locations
  without echoing input; unhandled errors return a generic 500 (traceback only in server
  logs).
- Every log line carries `request_id` (inbound `X-Request-ID` if well-formed, otherwise
  generated; echoed in the response) and the claimed `tenant_id` (only if it is a valid
  UUID). One access line per request: method, path (includes references such as
  `ORD-1001`), status, duration. Query strings and bodies are not logged.

## Seed data (synthetic)

`python -m scripts.seed_demo` — two tenants (Northstar Commerce / USD, BluePeak Retail /
EUR) that deliberately reuse `CUS-1001`, `ORD-1001`, `INV-1001`, `SHP-1001`, `SKU-1001`.
Rules: refuses `APP_ENV=production`; ids are UUID5 of the natural key; upserts conflict on
those ids only; it never deletes and never overwrites unrelated rows (a conflicting
non-demo natural key aborts the whole seed); unchanged rows are not rewritten. Dates are
fixed, so derived states such as "overdue" change as real time passes (by design).

## Open items

Known follow-ups are tracked in [pending-items.md](pending-items.md).

## Testing

- Unit tests run without a database.
- Real-PostgreSQL tests need `TEST_DATABASE_URL` naming a database ending in `_test`; the
  session fixture runs `alembic downgrade base` → `upgrade head` → seed, and each test runs
  in a rolled-back transaction.
- Authoritative isolation proof: composite-FK rejection tests, service-level and API-level
  cross-tenant tests, shared references across tenants. A SQL-capture test checking that
  every statement is bound to the tenant id is a supplementary regression guard only.
