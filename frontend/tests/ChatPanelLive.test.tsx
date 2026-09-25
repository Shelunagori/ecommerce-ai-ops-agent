import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import ChatPanel from "@/components/agent/ChatPanel";
import type { Membership } from "@/lib/types";

import {
  AGENT_CAPS,
  APPROVAL_TRACE,
  DECIDED_TRACE,
  NS,
  TOOLS_TRACE,
  controlledStream,
  mockFetchStreams,
  pendingAction,
  response,
  runEvents,
  sse,
} from "./fixtures";

const approver: Membership = { tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "approver" };
const EMPTY = { thread_id: "t-1", messages: [], pending_action: null };
const HISTORY = { path: "/api/agent/threads/t-1/messages", body: EMPTY };

async function send(text: string) {
  await userEvent.type(screen.getByLabelText("Message"), text);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

const panel = () => screen.getByTestId("trace-panel");
const panelRow = (kind: string, nth = 0) =>
  within(panel()).getAllByTestId("trace-step").filter((r) => r.dataset.kind === kind)[nth];

function toolRun() {
  const e = runEvents();
  return {
    started: sse(
      e("run_started", { capabilities: AGENT_CAPS }),
      e("step_started", { step_id: "s1", kind: "request", label: "Request received", status: "running" }),
      e("step_completed", { step_id: "s1", kind: "request", label: "Request received", status: "completed", duration_ms: 1 }),
      e("step_started", { step_id: "s2", kind: "model", label: "Agent orchestration", status: "running", metadata: { call: 1, provider: "gemini" } }),
    ),
    tool: sse(
      e("step_completed", { step_id: "s2", kind: "model", label: "Agent orchestration", status: "completed", detail: "Requested commerce tool: get_shipment", metadata: { call: 1, provider: "gemini" }, duration_ms: 700 }),
      e("step_started", { step_id: "s3", kind: "commerce_tool", label: "Tool: get_shipment", status: "running", metadata: { tool: "get_shipment" } }),
    ),
    rest: sse(
      e("step_completed", { step_id: "s3", kind: "commerce_tool", label: "Tool: get_shipment", status: "completed", metadata: { tool: "get_shipment", outcome: "success", duration_ms: 4.2 }, duration_ms: 4.2 }),
      e("step_started", { step_id: "s4", kind: "model", label: "Agent orchestration", status: "running" }),
      e("step_completed", { step_id: "s4", kind: "model", label: "Agent orchestration", status: "completed", detail: "Composed the final answer", duration_ms: 600 }),
      e("step_completed", { step_id: "s5", kind: "response", label: "Response generated", status: "completed" }),
      e("step_skipped", { step_id: "s6", kind: "retrieval", label: "Policy retrieval", status: "skipped", detail: "Not used in this run" }),
      e("step_skipped", { step_id: "s7", kind: "grounding", label: "Grounding validation", status: "skipped", detail: "Not used in this run" }),
      e("step_skipped", { step_id: "s8", kind: "action_proposal", label: "Action proposal", status: "skipped", detail: "Not used in this run" }),
      e("run_completed", { response: response({ answer: "SHP-1003 is delayed.", execution_trace: TOOLS_TRACE, tool_calls: [{ round: 1, tool: "get_shipment", arguments: {}, rejected_argument_names: [], outcome: "success", error_code: null, duration_ms: 4.2 }] }) }),
    ),
  };
}

describe("ChatPanel live execution trace", () => {
  it("the trace appears the moment Send is clicked and follows real events to the answer", async () => {
    const stream = controlledStream();
    const calls = mockFetchStreams([HISTORY, { method: "POST", path: "/api/agent/messages/stream", sse: stream }]);
    const run = toolRun();
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Where is SHP-1003?");
    // before any backend event: transport state in the panel and a working placeholder in chat
    expect(within(panel()).getAllByTestId("transport-step")[0]).toHaveTextContent("Sending request…");
    expect(screen.getByTestId("working")).toHaveTextContent("CommerceOps AI is working…");

    stream.push(run.started);
    await waitFor(() => expect(panelRow("model").dataset.status).toBe("running"));
    expect(panelRow("request").dataset.status).toBe("completed");
    expect(screen.getByTestId("current-step")).toHaveTextContent("Current step: Agent orchestration");

    stream.push(run.tool);
    await waitFor(() => expect(panelRow("commerce_tool").dataset.status).toBe("running"));
    expect(panelRow("model").dataset.status).toBe("completed");
    expect(screen.getByTestId("current-step")).toHaveTextContent("Current step: Tool: get_shipment");

    stream.push(run.rest);
    stream.close();
    const msg = await screen.findByText("SHP-1003 is delayed.");
    expect(msg).toBeInTheDocument();
    expect(screen.queryByTestId("working")).toBeNull();
    // converged on the final trace + skipped capabilities; exactly one answer bubble
    expect(within(panel()).getAllByTestId("trace-step").map((r) => `${r.dataset.kind}:${r.dataset.status}`)).toEqual([
      "request:completed",
      "model:completed",
      "commerce_tool:completed",
      "model:completed",
      "response:completed",
      "retrieval:skipped",
      "grounding:skipped",
      "action_proposal:skipped",
    ]);
    expect(within(panel()).getByTestId("run-summary")).toHaveTextContent("12ms · 2 model calls · 1 tool");
    expect(screen.getAllByTestId("assistant-message")).toHaveLength(1);
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.url).toMatch(/\/api\/agent\/messages\/stream$/);
    expect(post.body).toEqual({ text: "Where is SHP-1003?", thread_id: "t-1" });
    expect(post.headers["X-Tenant-ID"]).toBe(NS);
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("run_failed keeps the red trace visible and uses the standard retryable error", async () => {
    const e = runEvents();
    mockFetchStreams([
      HISTORY,
      {
        method: "POST",
        path: "/api/agent/messages/stream",
        sse: sse(
          e("run_started", { capabilities: AGENT_CAPS }),
          e("step_completed", { step_id: "s1", kind: "request", label: "Request received", status: "completed" }),
          e("step_started", { step_id: "s2", kind: "retrieval", label: "Policy retrieval", status: "running" }),
          e("step_failed", { step_id: "s2", kind: "retrieval", label: "Policy retrieval", status: "failed", detail: "Policy retrieval service was unavailable" }),
          e("step_skipped", { step_id: "s3", kind: "response", label: "Response generated", status: "skipped", detail: "Not run: the run stopped" }),
          e("run_failed", { error: { code: "agent_retrieval_error", message: "Policy knowledge could not be retrieved.", status: 503 } }),
        ),
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Policy?");
    await waitFor(() => expect(panelRow("retrieval").dataset.status).toBe("failed"));
    expect(panelRow("retrieval")).toHaveTextContent("Policy retrieval service was unavailable");
    expect(panelRow("response").dataset.status).toBe("skipped");
    expect(screen.getByTestId("run-failed")).toBeInTheDocument();
    const alerts = screen.getAllByRole("alert").map((a) => a.textContent);
    expect(alerts.some((t) => t?.includes("Policy knowledge could not be retrieved."))).toBe(true);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("mobile: the running answer offers 'View live execution' inline", async () => {
    const stream = controlledStream();
    mockFetchStreams([HISTORY, { method: "POST", path: "/api/agent/messages/stream", sse: stream }]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Where is SHP-1003?");
    stream.push(toolRun().started);
    const msg = await screen.findByTestId("assistant-message");
    await userEvent.click(within(msg).getByRole("button", { name: "View live execution" }));
    const inline = within(msg).getByTestId("inline-trace");
    const model = () => within(inline).getAllByTestId("trace-step").find((r) => r.dataset.kind === "model")!;
    await waitFor(() => expect(model().dataset.status).toBe("running"));
    expect(model()).toHaveTextContent("Running");
    expect(within(inline).getByTestId("run-summary")).toHaveTextContent(/^Running · /);
  });

  it("approve streams the resumed run: waiting -> execution running -> completed", async () => {
    const e = runEvents("c".repeat(32));
    const decision = controlledStream();
    const calls = mockFetchStreams([
      HISTORY,
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Needs approval.", action: pendingAction, execution_trace: APPROVAL_TRACE }) },
      { method: "POST", path: `/api/agent/actions/${pendingAction.id}/approve/stream`, sse: decision },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004"); // stream route not mocked -> the plain endpoint, once
    expect(await screen.findByTestId("action-status")).toHaveTextContent("Pending approval");
    expect(panelRow("approval").dataset.status).toBe("waiting");
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    decision.push(
      sse(
        e("run_started", { capabilities: [{ kind: "approval", label: "Human decision" }, { kind: "action_execution", label: "Deterministic execution" }, { kind: "response", label: "Response generated" }] }),
        e("approval_resolved", { step_id: "s1", kind: "approval", label: "Approved by a human", status: "completed", metadata: { action_type: "cancel_order", decision: "approved" } }),
        e("step_started", { step_id: "s2", kind: "action_execution", label: "Deterministic execution", status: "running" }),
      ),
    );
    await waitFor(() => expect(panelRow("action_execution").dataset.status).toBe("running"));
    expect(panelRow("approval").dataset.status).toBe("completed");
    expect(within(panel()).getAllByTestId("trace-step").map((r) => r.dataset.kind).slice(0, 3)).toEqual(["request", "model", "action_proposal"]);
    decision.push(
      sse(
        e("step_completed", { step_id: "s2", kind: "action_execution", label: "Deterministic execution", status: "completed", metadata: { audit_recorded: true } }),
        e("step_completed", { step_id: "s3", kind: "response", label: "Outcome recorded", status: "completed" }),
        e("run_completed", {
          response: {
            action: { ...pendingAction, status: "succeeded", evidence: [], created_at: "", decided_at: "", completed_at: "", events: [] },
            answer: "Done: order ORD-1004 was cancelled.",
            execution_trace: DECIDED_TRACE,
          },
        }),
      ),
    );
    decision.close();
    expect(await screen.findByText("Done: order ORD-1004 was cancelled.")).toBeInTheDocument();
    expect(screen.getByTestId("action-status")).toHaveTextContent("Succeeded");
    expect(panelRow("action_execution").dataset.status).toBe("completed");
    expect(within(panel()).getAllByTestId("trace-step").map((r) => r.dataset.kind)).toEqual(
      DECIDED_TRACE.map((t) => t.kind),
    );
    const approves = calls.filter((c) => c.url.includes("/approve"));
    expect(approves.map((c) => c.url.replace(/.*actions\//, ""))).toEqual([`${pendingAction.id}/approve/stream`]);
    expect(approves[0].body).toEqual({ arguments_hash: pendingAction.arguments_hash });
  });

  it("reject: the decision is recorded and execution is skipped", async () => {
    const e = runEvents("d".repeat(32));
    mockFetchStreams([
      HISTORY,
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Needs approval.", action: pendingAction, execution_trace: APPROVAL_TRACE }) },
      {
        method: "POST",
        path: /\/reject\/stream$/,
        sse: sse(
          e("run_started", { capabilities: [] }),
          e("approval_resolved", { step_id: "s1", kind: "approval", label: "Rejected by a human", status: "rejected", detail: "Nothing was changed" }),
          e("step_skipped", { step_id: "s2", kind: "action_execution", label: "Deterministic execution", status: "skipped", detail: "Not executed: nothing was changed" }),
          e("step_completed", { step_id: "s3", kind: "response", label: "Outcome recorded", status: "completed" }),
          e("run_completed", {
            response: {
              action: { ...pendingAction, status: "rejected", evidence: [], created_at: "", decided_at: "", completed_at: "", events: [] },
              answer: "The request was rejected; nothing was changed.",
              execution_trace: [...APPROVAL_TRACE.slice(0, 3), { sequence: 4, kind: "approval", label: "Rejected by a human", status: "rejected", detail: "Nothing was changed", metadata: {} }, { sequence: 5, kind: "response", label: "Outcome recorded", status: "completed", detail: null, metadata: {} }],
            },
          }),
        ),
      },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    await userEvent.click(await screen.findByRole("button", { name: "Reject" }));
    expect(await screen.findByText("The request was rejected; nothing was changed.")).toBeInTheDocument();
    expect(panelRow("approval").dataset.status).toBe("rejected");
    expect(panelRow("action_execution").dataset.status).toBe("skipped");
    expect(within(panel()).queryAllByTestId("trace-step").some((r) => r.dataset.kind === "action_execution" && r.dataset.status === "completed")).toBe(false);
  });

  it("a lost approve stream is never retried: one mutation request, a reload hint", async () => {
    const decision = controlledStream();
    const calls = mockFetchStreams([
      HISTORY,
      { method: "POST", path: "/api/agent/messages", body: response({ answer: "Needs approval.", action: pendingAction, execution_trace: APPROVAL_TRACE }) },
      { method: "POST", path: `/api/agent/actions/${pendingAction.id}/approve/stream`, sse: decision },
    ]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Cancel ORD-1004");
    await userEvent.click(await screen.findByRole("button", { name: "Approve" }));
    decision.push(sse(runEvents()("run_started", { capabilities: [] })));
    decision.error(); // network drops mid-run
    expect(await screen.findByText(/The decision may have been recorded/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload conversation" })).toBeInTheDocument();
    const mutations = calls.filter((c) => c.method === "POST" && c.url.includes("/actions/"));
    expect(mutations).toHaveLength(1); // no fallback, no automatic retry
  });

  it("an interrupted message is not resent", async () => {
    const stream = controlledStream();
    const calls = mockFetchStreams([HISTORY, { method: "POST", path: "/api/agent/messages/stream", sse: stream }]);
    render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Where is SHP-1003?");
    stream.push(toolRun().started);
    await waitFor(() => expect(panelRow("model").dataset.status).toBe("running"));
    stream.error();
    expect(await screen.findByText(/it was not sent again/)).toBeInTheDocument();
    expect(within(panel()).getByTestId("run-summary")).toHaveTextContent("Connection lost");
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("starting a new conversation stops listening to the previous run", async () => {
    const stream = controlledStream();
    const calls = mockFetchStreams([HISTORY, { method: "POST", path: "/api/agent/messages/stream", sse: stream }]);
    const { unmount } = render(<ChatPanel tenant={approver} threadId="t-1" />);
    await send("Where is SHP-1003?");
    stream.push(toolRun().started);
    await waitFor(() => expect(panelRow("model").dataset.status).toBe("running"));
    const post = calls.find((c) => c.method === "POST")!;
    unmount(); // AgentApp re-keys ChatPanel for a new conversation
    expect(post.signal?.aborted).toBe(true);
  });
});
