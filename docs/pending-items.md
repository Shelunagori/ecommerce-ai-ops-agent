# Pending items

Follow-ups found during development that are intentionally **not** addressed yet.
Resolved modelling decisions live in [architecture.md](architecture.md), not here.

| # | Item | Context | Raised |
| --- | --- | --- | --- |
| P1 | Seed guard blocks a hosted demo seed | `scripts/seed_demo.py` refuses `APP_ENV=production`. Seeding a Supabase demo database from an environment configured as production is therefore blocked. | Step 2 |
| P2 | Explicit production demo-seed mechanism for Supabase | Decide how the public demo gets synthetic data (e.g. a deliberate, explicitly-flagged one-off command against the Supabase URL) without weakening the production guard in general. | Step 2 |
| P3 | Uvicorn logging consistency | Application logs are JSON lines; uvicorn's own startup/access logs are still plain text. Align them (or disable uvicorn's access log in favour of the app's request log). | Step 1 |
| P4 | Supabase transaction-pooler prepared statements | Supabase's transaction pooler (port 6543) needs psycopg prepared statements disabled (`prepare_threshold=None`). Today the docs recommend the direct / session-pooler connection instead. | Step 1 |
| P6 | Unbounded thread history in checkpoints | A continued LangGraph thread appends every message to its checkpoint and sends the full history to the model; nothing trims, summarises or caps it. Decide pruning/summarisation (and a size limit) together with durable persistence and conversation memory. | Step 6 |
| P7 | Scope-mismatch refusal still stores the intruding user message | If trusted code bypasses `CommerceGraphAssistant` and invokes the compiled graph with another tenant's thread key, the MODEL node refuses before any model call, but the input `HumanMessage` has already been appended to that thread by LangGraph's input step. Unreachable through the runner (keys are tenant-derived); decide whether to validate scope before input is written (e.g. an entry node or runner-side pre-check). | Step 6 |
| P8 | Retrieval evaluation growth and tuning | The Step-7 evaluation set has 20 cases over a 12-document synthetic corpus. Before judging lexical vs semantic retrieval (and chunk size / overlap), grow the corpus and cases (paraphrases, negatives, multi-section answers) so differences are not noise. | Step 7 |
