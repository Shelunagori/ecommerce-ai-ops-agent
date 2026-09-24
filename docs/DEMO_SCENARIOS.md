# Demo scenarios

Five short scenarios (all five are also automated in `frontend/e2e/agent.spec.ts`) that show what CommerceOps AI does and where its guarantees come from.
All data is synthetic (two invented tenants: **Northstar Commerce** and **BluePeak Retail**).

## Two ways to run them

**A. Offline, deterministic (no LLM, no keys).** The real API, database, tools, retrieval,
approvals and checkpoints, with a keyword-scripted chat model — the same harness the browser
e2e tests use. Needs a dedicated `*_test` database (it is reset on start).

```bash
# terminal 1
cd backend
TEST_DATABASE_URL=postgresql://commerceops:<password>@localhost:5432/commerceops_test \
  uv run python -m tests.e2e.server --port 8100 --origin http://127.0.0.1:3100
# terminal 2
cd frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8100 NEXT_PUBLIC_AUTH_MODE=demo npx next dev -H 127.0.0.1 -p 3100
# open http://127.0.0.1:3100
```

**B. Live local model.** The production runtime with Ollama (see `docs/DEVELOPMENT.md`):
`alembic upgrade head`, `seed_demo`, `ingest_policies`, `embed_policies`,
`setup_checkpoints`, then `uvicorn app.main:app` and `npm run dev`. A small local model
phrases answers differently and can make mistakes; the guarantees below do not depend on
the model.

The wording in the boxes is for mode A (the keyword model). In mode B ask the same thing in
your own words.

---

## 1. Exact business facts come from the database, not from RAG

> Tenant **Northstar** — "Show me order ORD-1001"

* Answer: *Order ORD-1001 is delivered.* The activity strip shows **Order lookup**.
* Switch the tenant to **BluePeak** and ask again: a *different* ORD-1001 (each tenant has its
  own). The tenant comes from the trusted principal, never from the model or the prompt.

**Why it matters:** order status, totals and dates are exact relational facts read through
tenant-scoped queries; the model cannot pass a tenant id or SQL.

## 2. Policy answers with validated citations

> Tenant **BluePeak** — "What compensation applies to a delayed shipment?"

* The answer carries a numbered citation; expanding the source card shows the policy title,
  version, section, effective dates and the `policy://…` citation.
* Citations are checked deterministically: only chunks retrieved in the **current** turn, for
  this tenant, effective on the requested date, are accepted.

**Measured (retrieval only):** On the fixed 20-case synthetic retrieval evaluation corpus,
semantic retrieval achieved exact-chunk hit@1 of 1.00 versus 0.40 for lexical retrieval.
(A small synthetic set: a regression signal, not a general accuracy claim.)

## 3. Mixed: a shipment fact plus the policy that applies

> Tenant **Northstar** — "Where is SHP-1003, and what compensation applies if it is delayed?"

* The activity strip shows **Shipment lookup** first, then **Policy search**: the status
  (*delayed*) comes from the database, the rule from a cited policy chunk.
* Order matters: once policy text is in the model's context, no new commerce tool may run in
  that turn (`commerce_call_after_retrieval`) — retrieved text cannot trigger capabilities.

## 4. Approval-gated cancellation

> Tenant **Northstar** — "Please cancel ORD-1004"

* The assistant only **proposes**: an approval card shows the exact arguments, the current
  order status, the expiry and the arguments hash. The order is still `processing`.
* **Approve** → *Succeeded*; the outcome message is written by the application (not the
  model): *Done: order ORD-1004 …* The order is now `cancelled`.
* Try "cancel ORD-1007" and **Reject**: *Rejected*, "nothing was changed"; ORD-1007 stays
  `confirmed`. Requests expire after `ACTION_APPROVAL_TTL_SECONDS` (default 15 min) and can
  then never execute. With `AUTH_MODE=supabase`, a `member` sees the card with
  approve/reject disabled (the API also returns 403 `action_forbidden`).

## 5. Approval-gated store credit with duplicate-execution protection

> Tenant **Northstar** — "Issue store credit to CUS-1002 for ORD-1003"

* The assistant first retrieves the delayed-shipment policy, then proposes a synthetic store
  credit of 5.00 USD whose **evidence** is the retrieved policy citation.
* The server re-checks at execution: customer ↔ order, currency, amount ≤ order total and ≤
  `STORE_CREDIT_MAX_AMOUNT`. Approve → exactly one row in the synthetic ledger, linked to the
  action request by an idempotency key. No real payment or refund provider exists.
* Approve the same request again (double click, second tab, `curl` from
  `docs/DEVELOPMENT.md`): the response carries the **same** ledger `transaction_id`, and the
  audit trail still has exactly one `action_succeeded`.
* Approval checks policy *evidence* exists and the business rules hold; it does not prove the
  policy was interpreted correctly — that is the approver's judgement.

---

### What to point out while demoing

* Every action and decision is in `audit_events`; every agent run in `agent_runs`
  (`docs/OBSERVABILITY.md`).
* The conversation survives an API restart: pending approvals live in PostgreSQL
  checkpoints and can be resumed by any instance.
* Nothing here is deployed; `docs/DEPLOYMENT.md` is the runbook.
