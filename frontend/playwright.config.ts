import { defineConfig, devices } from "@playwright/test";

/**
 * Browser end-to-end tests (Phase 10): the real UI against the real API.
 *
 * Starts two servers:
 *  - backend: `tests.e2e.server` — the real FastAPI app on a dedicated *_test PostgreSQL
 *    database (reset + seeded), real tools/retrieval/approvals/checkpoints, with a
 *    deterministic keyword chat model instead of an LLM (no network, no keys);
 *  - frontend: `next dev` in demo auth mode pointed at that backend.
 *
 * Requires TEST_DATABASE_URL (the same dedicated database the pytest suite uses — do not run
 * both at once) and `uv` for the backend. Run: `npm run test:e2e`.
 */
const API = "http://127.0.0.1:8100";
const WEB = "http://127.0.0.1:3100";
const backendPython = process.env.E2E_BACKEND_PYTHON ?? "uv run python";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: [["list"]],
  use: {
    baseURL: WEB,
    trace: "retain-on-failure",
    launchOptions: process.env.E2E_CHROMIUM_PATH ? { executablePath: process.env.E2E_CHROMIUM_PATH } : {},
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `${backendPython} -m tests.e2e.server --port 8100 --origin ${WEB}`,
      cwd: "../backend",
      url: `${API}/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      stdout: "ignore",
      stderr: "pipe",
      // SIGTERM (not the default SIGKILL) so the harness's shutdown drops its checkpoint
      // tables and leaves the *_test database as the pytest migration tests expect.
      gracefulShutdown: { signal: "SIGTERM", timeout: 10_000 },
    },
    {
      // E2E_WEB_COMMAND="npx next build && npx next start -H 127.0.0.1 -p 3100" runs the production build.
      command: process.env.E2E_WEB_COMMAND ?? "npx next dev --hostname 127.0.0.1 --port 3100",
      url: WEB,
      timeout: 180_000,
      reuseExistingServer: false,
      env: { NEXT_PUBLIC_API_URL: API, NEXT_PUBLIC_AUTH_MODE: "demo" },
    },
  ],
});
