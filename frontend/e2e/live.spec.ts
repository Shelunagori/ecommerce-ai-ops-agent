import { expect, test, type Page } from "@playwright/test";

/**
 * Live execution trace against the REAL backend (e2e harness: real API + PostgreSQL + real
 * tools/retrieval/actions, keyword model). The harness paces each real model call, tool,
 * retrieval and action execution (`/__e2e/pacing`) so every step is observable while it runs;
 * nothing in the browser simulates progress.
 *
 * A MutationObserver records every status each trace row takes in the side panel, so the
 * assertions check the whole running -> completed/failed history, not a lucky screenshot.
 */
const API = "http://127.0.0.1:8100";
type Membership = { tenant_id: string; slug: string };
type Seen = { kind: string; status: string; amber: boolean; green: boolean; red: boolean };
let tenants: Record<string, Membership>;

test.beforeAll(async ({ request }) => {
  const me = await (await request.get(`${API}/api/me`)).json();
  tenants = Object.fromEntries(me.memberships.map((m: Membership) => [m.slug, m]));
});

test.beforeEach(async ({ page, request }) => {
  expect((await request.post(`${API}/__e2e/reset`)).ok()).toBeTruthy();
  await page.addInitScript(() => {
    const log: Array<Record<string, unknown>> = [];
    (window as unknown as { __traceLog: typeof log }).__traceLog = log;
    const record = (el: Element) => {
      if (!(el instanceof HTMLElement) || el.dataset.testid !== "trace-step") return;
      if (!el.closest('[data-testid="trace-panel"]')) return;
      const icon = el.querySelector("span[aria-hidden]:not(.absolute)")?.className ?? "";
      log.push({
        kind: el.dataset.kind,
        status: el.dataset.status,
        amber: el.className.includes("amber"),
        green: icon.includes("emerald"),
        red: el.className.includes("rose"),
      });
    };
    new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === "attributes") record(m.target as Element);
        m.addedNodes.forEach((n) => {
          if (n instanceof HTMLElement) {
            record(n);
            n.querySelectorAll('[data-testid="trace-step"]').forEach(record);
          }
        });
      }
    }).observe(document, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-status"] });
  });
});

async function pacing(page: Page, delay_ms: number) {
  expect((await page.request.post(`${API}/__e2e/pacing`, { data: { delay_ms } })).ok()).toBeTruthy();
}

async function open(page: Page, slug: string) {
  await page.goto("/");
  await page.getByLabel("Tenant").selectOption(tenants[slug].tenant_id);
  await expect(page.getByPlaceholder("Ask CommerceOps AI…")).toBeVisible();
}

async function send(page: Page, text: string) {
  await page.getByPlaceholder("Ask CommerceOps AI…").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
}

const log = (page: Page) => page.evaluate(() => (window as unknown as { __traceLog: Seen[] }).__traceLog);

function history(seen: Seen[], kind: string): string[] {
  const out: string[] = [];
  for (const s of seen.filter((x) => x.kind === kind)) if (out[out.length - 1] !== s.status) out.push(s.status);
  return out;
}

test.describe("desktop", () => {
  test.use({ viewport: { width: 1400, height: 900 } });

  test("mixed query: each real step is amber while running, then green", async ({ page }) => {
    await pacing(page, 450);
    await open(page, "northstar-commerce");
    await send(page, "Where is SHP-1003, and what compensation applies if it is delayed?");
    const panel = page.getByTestId("trace-panel");
    // The panel shows the run immediately, before the first model call has finished.
    await expect(panel.locator('[data-kind="model"][data-status="running"]').first()).toBeVisible();
    await expect(page.getByTestId("working")).toContainText("CommerceOps AI is working…");
    await expect(panel.getByTestId("run-summary")).toContainText("Running ·");

    const answer = page.getByTestId("assistant-message").last();
    await expect(answer).toContainText("Shipment SHP-1003 is delayed.", { timeout: 20_000 });
    await expect(answer.getByRole("link", { name: /Source 1: Delayed Shipment Compensation Policy/ })).toBeVisible();

    const seen = await log(page);
    for (const kind of ["model", "commerce_tool", "retrieval"]) {
      expect(history(seen, kind).slice(0, 2), kind).toEqual(["running", "completed"]);
      expect(seen.some((s) => s.kind === kind && s.status === "running" && s.amber), `${kind} amber`).toBe(true);
      expect(seen.some((s) => s.kind === kind && s.status === "completed" && s.green), `${kind} green`).toBe(true);
    }
    expect(history(seen, "request").at(-1)).toBe("completed");
    expect(history(seen, "grounding")).toContain("completed");
    expect(history(seen, "response").at(-1)).toBe("completed");
    expect(seen.some((s) => s.status === "failed")).toBe(false);

    const rows = panel.locator('[data-testid="trace-step"]:not([data-status="skipped"])');
    await expect(rows).toHaveCount(8);
    expect(await rows.evaluateAll((els) => els.map((e) => `${e.getAttribute("data-kind")}:${e.getAttribute("data-status")}`))).toEqual([
      "request:completed",
      "model:completed",
      "commerce_tool:completed",
      "model:completed",
      "retrieval:completed",
      "model:completed",
      "grounding:completed",
      "response:completed",
    ]);
    await expect(panel.locator('[data-kind="action_proposal"][data-status="skipped"]')).toContainText("Not used");
    await expect(panel.getByTestId("run-summary")).toContainText("3 model calls · 1 tool");
  });

  test("controlled retrieval failure: the failed step turns red and later steps are skipped", async ({ page }) => {
    await pacing(page, 250);
    expect((await page.request.post(`${API}/__e2e/faults`, { data: { retrieval: true } })).ok()).toBeTruthy();
    await open(page, "bluepeak-retail");
    await send(page, "What compensation applies to a delayed shipment?");
    const panel = page.getByTestId("trace-panel");
    const failed = panel.locator('[data-kind="retrieval"][data-status="failed"]');
    await expect(failed).toBeVisible({ timeout: 20_000 });
    await expect(failed).toContainText("Policy retrieval service was unavailable");
    await expect(failed).toHaveAttribute("role", "alert");
    await expect(panel.locator('[data-kind="grounding"][data-status="skipped"]')).toBeVisible();
    await expect(panel.locator('[data-kind="response"][data-status="skipped"]')).toBeVisible();
    await expect(panel.getByTestId("run-summary")).toContainText("Failed after");
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
    const seen = await log(page);
    expect(history(seen, "retrieval")).toEqual(["running", "failed"]);
    expect(seen.some((s) => s.kind === "retrieval" && s.status === "failed" && s.red)).toBe(true);
    expect(seen.some((s) => s.kind === "response" && s.status === "completed")).toBe(false);
  });

  test("HITL: the trace pauses on approval, then execution runs live after Approve", async ({ page }) => {
    await pacing(page, 400);
    await open(page, "northstar-commerce");
    await send(page, "Please cancel ORD-1004");
    const panel = page.getByTestId("trace-panel");
    const approval = panel.locator('[data-kind="approval"][data-status="waiting"]');
    await expect(approval).toBeVisible({ timeout: 20_000 });
    await expect(approval).toContainText("HUMAN APPROVAL REQUIRED");
    await expect(approval.getByTestId("running-dot")).toHaveCount(0); // a stable wait, not a spinner
    await expect(panel.getByTestId("run-summary")).toHaveText("Waiting for human approval");
    await expect(panel.locator('[data-testid="trace-placeholder"][data-kind="action_execution"]')).toBeVisible();

    await page.getByRole("button", { name: "Approve" }).click();
    await expect(page.getByTestId("assistant-message").last()).toContainText("Done: order ORD-1004", { timeout: 20_000 });
    const seen = await log(page);
    expect(history(seen, "action_execution").slice(0, 2)).toEqual(["running", "completed"]);
    await expect(panel.locator('[data-kind="approval"]').first()).toHaveAttribute("data-status", "completed");
    await expect(panel.locator('[data-kind="action_execution"]')).toHaveAttribute("data-status", "completed");
    await expect(panel.locator('[data-kind="action_execution"]')).toContainText("audit event recorded");
    await expect(panel.locator('[data-kind="response"]')).toContainText("Outcome recorded");
  });
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("the live trace opens inline, shows running state as text, and does not overflow", async ({ page }) => {
    await pacing(page, 600);
    await open(page, "northstar-commerce");
    await send(page, "Where is SHP-1003?");
    await expect(page.getByTestId("trace-panel")).toBeHidden();
    const answer = page.getByTestId("assistant-message").last();
    await answer.getByRole("button", { name: "View live execution" }).click();
    const inline = answer.getByTestId("inline-trace");
    const running = inline.locator('[data-testid="trace-step"][data-status="running"]').first();
    await expect(running).toBeVisible();
    await expect(running).toContainText("Running"); // not color alone
    await expect(answer).toContainText("Shipment SHP-1003 is delayed.", { timeout: 20_000 });
    await expect(inline.locator('[data-kind="response"]')).toHaveAttribute("data-status", "completed");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
