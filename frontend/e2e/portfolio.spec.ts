import { expect, test, type Page } from "@playwright/test";

/**
 * Portfolio experience: the execution trace rendered from the REAL backend trace, and the
 * public /review page. Same harness as agent.spec.ts (real API + PostgreSQL, keyword model).
 */
const API = "http://127.0.0.1:8100";
type Membership = { tenant_id: string; slug: string };
let tenants: Record<string, Membership>;

test.beforeAll(async ({ request }) => {
  const me = await (await request.get(`${API}/api/me`)).json();
  tenants = Object.fromEntries(me.memberships.map((m: Membership) => [m.slug, m]));
});

test.beforeEach(async ({ request }) => {
  expect((await request.post(`${API}/__e2e/reset`)).ok()).toBeTruthy();
});

async function open(page: Page, slug: string) {
  await page.goto("/");
  await page.getByLabel("Tenant").selectOption(tenants[slug].tenant_id);
  await expect(page.getByPlaceholder("Ask CommerceOps AI…")).toBeVisible();
}

async function ask(page: Page, text: string) {
  const before = await page.getByTestId("assistant-message").count();
  await page.getByPlaceholder("Ask CommerceOps AI…").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByTestId("assistant-message")).toHaveCount(before + 1);
  return page.getByTestId("assistant-message").nth(before);
}

const kinds = (page: Page) =>
  // executed steps only (capabilities the run did not use are listed as skipped after it)
  page
    .getByTestId("trace-panel")
    .locator('[data-testid="trace-step"]:not([data-status="skipped"])')
    .evaluateAll((els) => els.map((e) => e.getAttribute("data-kind")));

test.describe("desktop", () => {
  test.use({ viewport: { width: 1400, height: 900 } });

  test("commerce lookup: trace shows the tool path from the real backend", async ({ page }) => {
    await open(page, "northstar-commerce");
    await expect(await ask(page, "Where is SHP-1003?")).toContainText("Shipment SHP-1003 is delayed.");
    await expect.poll(() => kinds(page)).toEqual(["request", "model", "commerce_tool", "model", "response"]);
    const panel = page.getByTestId("trace-panel");
    await expect(panel).toContainText("Tool: get_shipment");
    await expect(panel.getByText("LangChain Tool")).toBeVisible();
    await expect(panel.getByText("pgvector")).toHaveCount(0);
  });

  test("policy RAG: retrieval + grounding; the harness's lexical retriever is labelled honestly", async ({ page }) => {
    await open(page, "bluepeak-retail");
    await ask(page, "What compensation applies to a delayed shipment?");
    await expect.poll(() => kinds(page)).toEqual(["request", "model", "retrieval", "model", "grounding", "response"]);
    const panel = page.getByTestId("trace-panel");
    await expect(panel.getByText("PostgreSQL full-text")).toBeVisible(); // harness uses lexical retrieval
    await expect(panel).toContainText("Grounding validation");
  });

  test("mixed: both capabilities in the executed order", async ({ page }) => {
    await open(page, "northstar-commerce");
    await ask(page, "Where is SHP-1003, and what compensation applies if it is delayed?");
    await expect
      .poll(() => kinds(page))
      .toEqual(["request", "model", "commerce_tool", "model", "retrieval", "model", "grounding", "response"]);
  });

  test("approval: waiting step, then execution after approve", async ({ page }) => {
    await open(page, "northstar-commerce");
    const answer = await ask(page, "Please cancel ORD-1004");
    const panel = page.getByTestId("trace-panel");
    await expect(panel.locator('[data-kind="approval"]')).toHaveAttribute("data-status", "waiting");
    await expect(panel).toContainText("HUMAN APPROVAL REQUIRED");
    await answer.getByRole("button", { name: "Approve" }).click();
    await expect(panel.locator('[data-kind="action_execution"]')).toHaveAttribute("data-status", "completed");
    await expect(panel).toContainText("Approved by a human");
  });

  test("the trace panel collapses", async ({ page }) => {
    await open(page, "northstar-commerce");
    await page.getByRole("button", { name: "Hide execution trace" }).click();
    await expect(page.getByTestId("trace-panel")).toHaveCount(0);
    await page.getByRole("button", { name: "Show execution trace" }).click();
    await expect(page.getByTestId("trace-panel")).toBeVisible();
  });
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("no side panel; the trace opens inline per answer", async ({ page }) => {
    await open(page, "northstar-commerce");
    const answer = await ask(page, "Where is SHP-1003?");
    await expect(page.getByTestId("trace-panel")).toBeHidden();
    await answer.getByRole("button", { name: "View execution trace" }).click();
    await expect(answer.getByTestId("inline-trace")).toContainText("Tool: get_shipment");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });

  test("/review has no horizontal overflow", async ({ page }) => {
    await page.goto("/review");
    await expect(page.getByTestId("architecture-diagram")).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});

test("/review is public and its walkthroughs work", async ({ page }) => {
  await page.goto("/review");
  await expect(page.getByRole("heading", { level: 1, name: "CommerceOps AI" })).toBeVisible();
  await page.getByRole("tab", { name: "HITL action" }).click();
  await expect(page.getByText("HUMAN APPROVAL REQUIRED")).toBeVisible();
  await expect(page.getByRole("link", { name: "Try Live Demo" }).first()).toHaveAttribute("href", "/?demo=1");
  await page.getByRole("link", { name: "Open Application" }).click();
  await expect(page.getByPlaceholder("Ask CommerceOps AI…")).toBeVisible();
});
