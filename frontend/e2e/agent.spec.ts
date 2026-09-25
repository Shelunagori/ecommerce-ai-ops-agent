import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

/**
 * Real browser → real Next.js UI → real FastAPI → real PostgreSQL. Only the chat model is
 * deterministic (see backend/tests/e2e/keyword_model.py). Demo auth mode.
 */
const API = "http://127.0.0.1:8100";
type Membership = { tenant_id: string; slug: string; name: string; role: string };

let tenants: Record<string, Membership>;

test.beforeAll(async ({ request }) => {
  const me = await (await request.get(`${API}/api/me`)).json();
  tenants = Object.fromEntries(me.memberships.map((m: Membership) => [m.slug, m]));
});

let cspViolations: string[] = [];

test.beforeEach(async ({ request, page }) => {
  expect((await request.post(`${API}/__e2e/reset`)).ok()).toBeTruthy();
  cspViolations = [];
  page.on("console", (m) => {
    if (m.type() === "error" && /Content Security Policy/i.test(m.text())) cspViolations.push(m.text());
  });
});

test.afterEach(() => {
  expect(cspViolations, "the security headers must not break the app").toEqual([]);
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
  return page.getByTestId("assistant-message").nth(before); // stable: later replies append below
}

async function orderStatus(request: APIRequestContext, slug: string, number: string) {
  const r = await request.get(`${API}/api/orders/${number}`, { headers: { "X-Tenant-ID": tenants[slug].tenant_id } });
  return (await r.json()).status as string;
}

test("order lookup shows the answer and the tool activity", async ({ page }) => {
  await open(page, "northstar-commerce");
  const answer = await ask(page, "Show me order ORD-1001");
  await expect(answer).toContainText("Order ORD-1001 is delivered.");
  await expect(answer.getByTestId("activity")).toContainText("Order lookup");
});

test("policy answer renders a numbered citation and an expandable source card", async ({ page }) => {
  await open(page, "bluepeak-retail");
  const answer = await ask(page, "What compensation applies to a delayed shipment?");
  await expect(answer.getByRole("link", { name: /Source 1: Delayed Shipment Compensation Policy/ })).toBeVisible();
  const card = answer.getByTestId("source-card").first();
  await card.getByRole("button").click();
  await expect(card).toContainText("policy://delayed-shipment-compensation/v2#chunk-");
  await expect(answer).not.toContainText("[policy://"); // raw markers are replaced
});

test("mixed question: shipment fact from the database, then a cited policy", async ({ page }) => {
  await open(page, "northstar-commerce");
  const answer = await ask(page, "Where is SHP-1003, and what compensation applies if it is delayed?");
  await expect(answer).toContainText("Shipment SHP-1003 is delayed");
  await expect(answer.getByRole("link", { name: /Source 1: Delayed Shipment Compensation Policy/ })).toBeVisible();
  const activity = answer.getByTestId("activity");
  await expect(activity).toContainText("Shipment lookup");
  await expect(activity).toContainText(/polic/i);
});

test("cancel → approval card → approve changes the order exactly once", async ({ page, request }) => {
  await open(page, "northstar-commerce");
  const answer = await ask(page, "Please cancel ORD-1004");
  const card = answer.getByTestId("approval-card");
  await expect(card.getByTestId("action-status")).toHaveText("Pending approval");
  await expect(card).toContainText("ORD-1004");
  expect(await orderStatus(request, "northstar-commerce", "ORD-1004")).toBe("processing"); // nothing yet

  const listed = await request.get(`${API}/api/agent/actions?status=pending_approval`, {
    headers: { "X-Tenant-ID": tenants["northstar-commerce"].tenant_id },
  });
  const [action] = await listed.json();

  await card.getByRole("button", { name: "Approve" }).click();
  await expect(card.getByTestId("action-status")).toHaveText("Succeeded");
  await expect(page.getByTestId("assistant-message").last()).toContainText("Done: order ORD-1004");
  expect(await orderStatus(request, "northstar-commerce", "ORD-1004")).toBe("cancelled");

  // A duplicate approve (double click, second tab, retry) is idempotent: same result, no re-execution.
  const again = await request.post(`${API}/api/agent/actions/${action.id}/approve`, {
    headers: { "X-Tenant-ID": tenants["northstar-commerce"].tenant_id },
    data: { arguments_hash: action.arguments_hash },
  });
  expect(again.status()).toBe(200);
  expect((await again.json()).action.status).toBe("succeeded");
  const events = (
    await (
      await request.get(`${API}/api/agent/actions/${action.id}`, {
        headers: { "X-Tenant-ID": tenants["northstar-commerce"].tenant_id },
      })
    ).json()
  ).events.map((e: { event_type: string }) => e.event_type);
  expect(events.filter((t: string) => t === "action_succeeded")).toHaveLength(1);

  // The other tenant cannot see or decide it.
  const foreign = await request.post(`${API}/api/agent/actions/${action.id}/approve`, {
    headers: { "X-Tenant-ID": tenants["bluepeak-retail"].tenant_id },
    data: { arguments_hash: action.arguments_hash },
  });
  expect(foreign.status()).toBe(404);
});

test("reject leaves the order unchanged", async ({ page, request }) => {
  await open(page, "northstar-commerce");
  const card = (await ask(page, "cancel ORD-1007")).getByTestId("approval-card");
  await card.getByRole("button", { name: "Reject" }).click();
  await expect(card.getByTestId("action-status")).toHaveText("Rejected");
  await expect(page.getByTestId("assistant-message").last()).toContainText("nothing was changed");
  expect(await orderStatus(request, "northstar-commerce", "ORD-1007")).toBe("confirmed");
});

test("store credit carries policy evidence and posts one ledger entry", async ({ page, request }) => {
  await open(page, "northstar-commerce");
  const card = (await ask(page, "Issue store credit to CUS-1002 for ORD-1003")).getByTestId("approval-card");
  await expect(card).toContainText("5.00");
  await expect(card).toContainText("policy://");
  await card.getByRole("button", { name: "Approve" }).click();
  await expect(card.getByTestId("action-status")).toHaveText("Succeeded");
  const headers = { "X-Tenant-ID": tenants["northstar-commerce"].tenant_id };
  const [done] = await (await request.get(`${API}/api/agent/actions?status=succeeded`, { headers })).json();
  expect(done.action_type).toBe("issue_store_credit");
  expect(done.evidence[0].citation).toMatch(/^policy:\/\//);
  expect(done.result.transaction_id).toBeTruthy(); // one synthetic ledger row, linked to the request

  // Duplicate execution protection: a second approve returns the SAME ledger row.
  const again = await request.post(`${API}/api/agent/actions/${done.id}/approve`, {
    headers,
    data: { arguments_hash: done.arguments_hash },
  });
  expect(again.status()).toBe(200);
  expect((await again.json()).action.result.transaction_id).toBe(done.result.transaction_id);
  const detail = await (await request.get(`${API}/api/agent/actions/${done.id}`, { headers })).json();
  expect(detail.events.filter((e: { event_type: string }) => e.event_type === "action_succeeded")).toHaveLength(1);
});

test("tenants keep separate conversations; switching back restores history", async ({ page }) => {
  await open(page, "northstar-commerce");
  await ask(page, "hello");
  await page.getByLabel("Tenant").selectOption(tenants["bluepeak-retail"].tenant_id);
  await expect(page.getByTestId("assistant-message")).toHaveCount(0);
  await page.getByLabel("Tenant").selectOption(tenants["northstar-commerce"].tenant_id);
  await expect(page.getByTestId("assistant-message")).toHaveCount(1); // restored from the API
  await expect(page.getByText("hello", { exact: true })).toBeVisible();
});

test("a failed request shows a retryable error and retry reaches the backend", async ({ page }) => {
  await open(page, "northstar-commerce");
  let failures = 1;
  // The UI streams the run (POST /api/agent/messages/stream); fail that request once.
  await page.route("**/api/agent/messages/stream", (route) => (failures-- > 0 ? route.abort("connectionrefused") : route.continue()));
  await page.getByPlaceholder("Ask CommerceOps AI…").fill("hello");
  await page.getByRole("button", { name: "Send" }).click();
  const alert = page.getByRole("alert").filter({ hasText: "unreachable" });
  await expect(alert).toBeVisible();
  await alert.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByTestId("assistant-message")).toHaveCount(1);
  await expect(page.getByTestId("assistant-message")).toContainText("Hello!");
});
