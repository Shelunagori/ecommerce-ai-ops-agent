import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import AgentApp from "@/components/agent/AgentApp";

import { NS, mockFetch } from "./fixtures";

describe("AgentApp (demo mode)", () => {
  it("offers only the tenants returned as memberships", async () => {
    mockFetch([
      { path: "/api/me", body: { subject: "demo-user", auth_mode: "demo", memberships: [{ tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "approver" }] } },
      { path: /\/messages$/, body: { thread_id: "x", messages: [], pending_action: null } },
    ]);
    render(<AgentApp />);
    const select = await screen.findByRole("combobox", { name: "Tenant" });
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toEqual(["Northstar Commerce (approver)"]);
    expect(screen.getByText(/Local demo mode/)).toBeInTheDocument();
  });

  it("authentication failure shows a sign-in prompt and retry", async () => {
    mockFetch([{ path: "/api/me", status: 401, body: { error: { code: "auth_invalid", message: "invalid" } } }]);
    render(<AgentApp />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Sign in again");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("an account without memberships sees no tenant", async () => {
    mockFetch([{ path: "/api/me", body: { subject: "u", auth_mode: "supabase", memberships: [] } }]);
    render(<AgentApp />);
    expect(await screen.findByText(/no tenant access/)).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });
});

describe("AgentApp public demo", () => {
  it("shows the read-only badge, only the demo tenant and the review link", async () => {
    mockFetch([
      {
        path: "/api/me",
        body: {
          subject: "anon-123",
          auth_mode: "supabase",
          public_demo: true,
          memberships: [{ tenant_id: "11a6d918-f89f-59b5-ab1e-45a109978c8a", slug: "bluepeak-retail", name: "BluePeak Retail", role: "member" }],
        },
      },
      { path: /\/messages$/, body: { thread_id: "x", messages: [], pending_action: null } },
    ]);
    render(<AgentApp />);
    expect(await screen.findByTestId("public-demo-badge")).toHaveTextContent("Public Demo · Read-only");
    const select = screen.getByRole("combobox", { name: "Tenant" });
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toEqual(["BluePeak Retail (member)"]);
    expect(screen.getByRole("link", { name: "Review / Architecture" })).toHaveAttribute("href", "/review");
  });

  it("a permanent account shows no demo badge", async () => {
    mockFetch([
      { path: "/api/me", body: { subject: "u", auth_mode: "supabase", public_demo: false, memberships: [{ tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "approver" }] } },
      { path: /\/messages$/, body: { thread_id: "x", messages: [], pending_action: null } },
    ]);
    render(<AgentApp />);
    await screen.findByRole("combobox", { name: "Tenant" });
    expect(screen.queryByTestId("public-demo-badge")).not.toBeInTheDocument();
  });
});
