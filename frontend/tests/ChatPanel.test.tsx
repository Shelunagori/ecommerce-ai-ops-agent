import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import ChatPanel from "@/components/agent/ChatPanel";
import type { Membership } from "@/lib/types";

import { NS, V2, citation, mockFetch, pendingAction, response } from "./fixtures";

const approver: Membership = { tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "approver" };
const EMPTY = { thread_id: "t-1", messages: [], pending_action: null };

async function send(text: string) {
  await userEvent.type(screen.getByLabelText("Message"), text);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("ChatPanel", () => {
  it("sends a message with the tenant selector and renders a cited answer", async () => {
    const calls = mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      {
        method: "POST",
        path: "/api/agent/messages",
        body: response({
          answer: `Store credit is 15% [${V2}].`,
          citations: [citation],
          retrievals: [{ round: 1, as_of: "2026-09-01", result_count: 3, citations: [V2], outcome: "success", error_code: null, rejected_argument_names: [], duration_ms: 3 }],
          tool_calls: [{ round: 1, tool: "get_shipment", arguments: {}, rejected_argument_names: [], outcome: "success", error_code: null, duration_ms: 2 }],
        }),
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("What compensation applies?");
    const msg = await screen.findByTestId("assistant-message");
    expect(within(msg).getByRole("link", { name: /Source 1: Delayed Shipment Compensation Policy version 2/ })).toBeInTheDocument();
    expect(within(msg).getByTestId("activity")).toHaveTextContent("Shipment lookup");
    expect(within(msg).getByTestId("activity")).toHaveTextContent("3 sources as of 2026-09-01");
    await userEvent.click(within(msg).getByRole("button", { name: /Delayed Shipment Compensation Policy/ }));
    expect(within(msg).getByText(V2)).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.body).toEqual({ text: "What compensation applies?", thread_id: "t-1" });
    expect(post.headers["X-Tenant-ID"]).toBe(NS);
    expect(JSON.stringify(post.body)).not.toContain("tenant");
  });

  it("pending approval -> approve sends the shown hash and shows the outcome", async () => {
    const calls = mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "This action needs your approval…", action: pendingAction }) },
      {
        method: "POST",
        path: `/api/agent/actions/${pendingAction.id}/approve`,
        body: {
          action: { ...pendingAction, status: "succeeded", evidence: [], created_at: "", decided_at: "", completed_at: "", events: [], result: { order_number: "ORD-1004" } },
          answer: "Done: order ORD-1004 was cancelled.",
        },
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    expect(await screen.findByTestId("action-status")).toHaveTextContent("Pending approval");
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(screen.getByTestId("action-status")).toHaveTextContent("Succeeded"));
    expect(await screen.findByText("Done: order ORD-1004 was cancelled.")).toBeInTheDocument();
    const approve = calls.find((c) => c.url.endsWith("/approve"))!;
    expect(approve.body).toEqual({ arguments_hash: pendingAction.arguments_hash });
  });

  it("reject path", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ action: pendingAction }) },
      { method: "POST", path: /\/reject$/, body: { action: { ...pendingAction, status: "rejected", evidence: [], created_at: "", decided_at: "", completed_at: "", events: [] }, answer: "The request was rejected; nothing was changed." } },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    await userEvent.click(await screen.findByRole("button", { name: "Reject" }));
    await waitFor(() => expect(screen.getByTestId("action-status")).toHaveTextContent("Rejected"));
  });

  it("expired approval surfaces the server status", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ action: pendingAction }) },
      { method: "POST", path: /\/approve$/, body: { action: { ...pendingAction, status: "expired", evidence: [], created_at: "", decided_at: null, completed_at: "", events: [] }, answer: "The approval window expired; nothing was changed." } },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    await userEvent.click(await screen.findByRole("button", { name: "Approve" }));
    await waitFor(() => expect(screen.getByTestId("action-status")).toHaveTextContent("Expired"));
  });

  it("restores history and a pending approval for the thread", async () => {
    mockFetch([
      {
        path: "/api/agent/threads/t-1/messages",
        body: {
          thread_id: "t-1",
          messages: [{ id: "h1", role: "user", content: "Cancel ORD-1004" }],
          pending_action: { ...pendingAction, evidence: [], created_at: "", decided_at: null, completed_at: null, events: [] },
        },
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    expect(await screen.findByText("Cancel ORD-1004")).toBeInTheDocument();
    expect(await screen.findByTestId("approval-card")).toBeInTheDocument();
  });

  it("backend failure shows a retryable error", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", status: 503, body: { error: { code: "agent_retrieval_error", message: "Policy knowledge could not be retrieved." } } },
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Recovered." }) },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Policy?");
    expect(await screen.findByRole("alert")).toHaveTextContent("Policy knowledge could not be retrieved.");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Recovered.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("explains a pending approval conflict", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", status: 409, body: { error: { code: "agent_approval_pending", message: "waiting" } } },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("hello");
    expect(await screen.findByRole("alert")).toHaveTextContent("Decide on the pending approval");
  });

  it("members cannot approve", async () => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: EMPTY },
      { method: "POST", path: "/api/agent/messages", body: response({ action: pendingAction }) },
    ]);
    render(<ChatPanel tenant={{ ...approver, role: "member" }} threadId="t-1" />);
    await send("Cancel ORD-1004");
    expect(await screen.findByRole("button", { name: "Approve" })).toBeDisabled();
  });
});
