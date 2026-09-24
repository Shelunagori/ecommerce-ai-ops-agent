# Observability and audit

No paid service is required. Three layers, all free of prompts, model output,
chain-of-thought, retrieved policy text, vectors and secrets:

| Layer | Where | Guarantees |
| --- | --- | --- |
| Structured JSON logs | stdout (`app.agent.graph`, `app.agent.llm`, `app.agent.tools`, `app.knowledge.retrieval`, `app.actions`, `app.api`) | one line per run / model call / tool call / retrieval / action step; ids, counts, codes, durations |
| `audit_events` | PostgreSQL (migration 0005) | the action lifecycle, written in the SAME transaction as the state change |
| `agent_runs` | PostgreSQL (migration 0005) | one row per graph run or resume; best effort (never fails a run) |

## Correlation ids

| Id | Where it appears |
| --- | --- |
| `request_id` | every log line of a request, `agent_runs.request_id`, `audit_events.request_id` |
| thread key (`cg1-<sha256(tenant:thread)>`) | `agent_runs.thread_key`, logs (`thread_key`, first 16 chars) — never the raw thread id |
| action request id | `action_requests.id`, `audit_events.action_request_id`, `agent_runs.action_request_id`, API responses, logs (`action_id`) |
| approval | the `approval_decided` / `approval_expired` audit event of an action request (the decision is a state of the request, not a separate object) |
| tool call id | `action_requests.tool_call_id`, `audit_events.tool_call_id` |

## Audit events

`action_requested` (actor `agent`) · `approval_decided` (actor = verified user subject,
`details.decision`) · `approval_expired` · `action_succeeded` · `action_failed`
(`details.failure_code`) · `action_duplicate_prevented` (an execute call on an already
succeeded request returned the original result). `details` holds only `status`,
`action_type`, `arguments_hash`, `decision`, `failure_code`.

## SLI-style measures (SQL over the tables)

```sql
-- agent success rate (last 24 h)
SELECT avg((outcome IN ('ok', 'approval_pending'))::int) FROM agent_runs
WHERE created_at > now() - interval '1 day';

-- grounding failures
SELECT count(*) FROM agent_runs WHERE grounding_failure;

-- approval rate / rejection rate
SELECT details->>'decision' AS decision, count(*) FROM audit_events
WHERE event_type = 'approval_decided' GROUP BY 1;

-- action success / failure rate by type
SELECT action_type, status, count(*) FROM action_requests GROUP BY 1, 2;

-- idempotent duplicate prevention count
SELECT count(*) FROM audit_events WHERE event_type = 'action_duplicate_prevented';

-- run latency percentiles
SELECT percentile_cont(ARRAY[0.5, 0.95]) WITHIN GROUP (ORDER BY duration_ms) FROM agent_runs;
```

Model, tool and retrieval latency come from the structured logs:
`app.agent.llm` → `duration_ms`, `attempts`; `app.agent.tools` → `tool`, `outcome`,
`duration_ms`; `app.knowledge.retrieval` → `retriever`, `result_count`, `duration_ms`.

## Optional LangSmith tracing

Off by default. It is enabled only when `LANGSMITH_TRACING=true` **and**
`LANGSMITH_API_KEY` are both set (`LANGSMITH_PROJECT` optional); otherwise the app sets
`LANGSMITH_TRACING=false` explicitly at startup. When enabled, LangSmith receives prompts and
model outputs — use it with synthetic data only. Core functionality never depends on it.

## Retention

Nothing is deleted automatically. `agent_runs` / `audit_events` grow with usage; prune by
`created_at` according to your policy (audit events reference action requests, so delete
audit events before the requests they reference). Checkpoint threads are deleted with
`delete_checkpoint_thread` (see `app/agent/graph/checkpoint.py`).
