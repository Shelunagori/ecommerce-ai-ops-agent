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
