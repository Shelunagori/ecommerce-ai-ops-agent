import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import ChatPanel from "@/components/agent/ChatPanel";
import type { Membership } from "@/lib/types";

import { APPROVAL_TRACE, DECIDED_TRACE, NS, TOOLS_TRACE, mockFetch, pendingAction, response } from "./fixtures";

const approver: Membership = { tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "approver" };
const member: Membership = { ...approver, role: "member" };
const EMPTY = { thread_id: "t-1", messages: [], pending_action: null };

async function send(text: string) {
  await userEvent.type(screen.getByLabelText("Message"), text);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("ChatPanel execution trace", () => {
  it("shows the run's trace in the side panel and inline on demand", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "SHP-1003 is delayed.", execution_trace: TOOLS_TRACE }) },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Where is SHP-1003?");
    const panel = await screen.findByTestId("trace-panel");
    expect(within(panel).getAllByTestId("trace-step").map((s) => s.dataset.kind)).toEqual([
      "request", "model", "commerce_tool", "model", "response",
    ]);
    const msg = screen.getByTestId("assistant-message");
    await userEvent.click(within(msg).getByRole("button", { name: "View execution trace" }));
    expect(within(msg).getByTestId("inline-trace")).toHaveTextContent("Tool: get_shipment");
  });

  it("the desktop panel is collapsible", async () => {
    mockFetch([{ path: "/api/agent/threads/t-1/messages", body: EMPTY }]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    const toggle = screen.getByRole("button", { name: "Hide execution trace" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(toggle);
    expect(screen.queryByTestId("trace-panel")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show execution trace" })).toHaveAttribute("aria-expanded", "false");
  });

  it("approval-required trace, then the resumed run's trace after approve", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Needs approval.", action: pendingAction, execution_trace: APPROVAL_TRACE }) },
      {
        method: "POST",
        path: `/api/agent/actions/${pendingAction.id}/approve`,
        body: {
          action: { ...pendingAction, status: "succeeded", evidence: [], created_at: "", decided_at: "", completed_at: "", events: [] },
          answer: "Done: order ORD-1004 was cancelled.",
          execution_trace: DECIDED_TRACE,
        },
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    const panel = await screen.findByTestId("trace-panel");
    const waiting = within(panel).getAllByTestId("trace-step").find((s) => s.dataset.kind === "approval")!;
    expect(waiting.dataset.status).toBe("waiting");
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    await waitFor(() =>
      expect(within(screen.getByTestId("trace-panel")).getAllByTestId("trace-step").map((s) => s.dataset.kind)).toContain("action_execution"),
    );
    const approval = within(screen.getByTestId("trace-panel")).getAllByTestId("trace-step").find((s) => s.dataset.kind === "approval")!;
    expect(approval).toHaveTextContent("Approved by a human");
  });

  it("older history has no trace and says so", async () => {
    mockFetch([
      {
        path: "/api/agent/threads/t-1/messages",
        body: { thread_id: "t-1", messages: [{ id: "1", role: "user", content: "hi" }, { id: "2", role: "assistant", content: "Hello" }], pending_action: null },
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await screen.findByText("Hello");
    expect(within(screen.getByTestId("trace-panel")).getByText("Send a message to see how the agent executes it.")).toBeInTheDocument();
  });

  it("read-only public demo: explains the restriction and cannot approve", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Needs approval.", action: pendingAction }) },
    ]);
    render(<ChatPanel tenant={member} threadId="t-1" readOnly />);
    expect(screen.getByText(/Public demo: read-only/)).toBeInTheDocument();
    await send("Cancel ORD-1004");
    expect(await screen.findByRole("button", { name: "Approve" })).toBeDisabled();
  });

  it("a read-only refusal from the API is shown in plain words", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", status: 403, body: { error: { code: "public_demo_read_only", message: "x" } } },
    ]);
    render(<ChatPanel tenant={member} threadId="t-1" readOnly />);
    await send("Cancel ORD-1004");
    expect(await screen.findByRole("alert")).toHaveTextContent("The public demo is read-only");
  });
});
