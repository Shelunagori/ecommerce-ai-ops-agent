# Pending items

Follow-ups found during development that are intentionally **not** addressed yet.
Resolved modelling decisions live in [architecture.md](architecture.md), not here.

| # | Item | Context | Raised |
| --- | --- | --- | --- |
| P1 | Seed guard blocks a hosted demo seed | `scripts/seed_demo.py` refuses `APP_ENV=production`. Seeding a Supabase demo database from an environment configured as production is therefore blocked. | Step 2 |
| P2 | Explicit production demo-seed mechanism for Supabase | Decide how the public demo gets synthetic data (e.g. a deliberate, explicitly-flagged one-off command against the Supabase URL) without weakening the production guard in general. | Step 2 |
| P3 | Uvicorn logging consistency | Application logs are JSON lines; uvicorn's own startup/access logs are still plain text. Align them (or disable uvicorn's access log in favour of the app's request log). | Step 1 |
| P4 | Supabase transaction-pooler prepared statements | Supabase's transaction pooler (port 6543) needs psycopg prepared statements disabled (`prepare_threshold=None`). Today the docs recommend the direct / session-pooler connection instead. | Step 1 |
