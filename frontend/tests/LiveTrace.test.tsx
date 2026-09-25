import { act, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import LiveTrace from "@/components/agent/trace/LiveTrace";
import { applyEvent, newRun, type LiveRun } from "@/lib/liveTrace";
import type { RunEvent } from "@/lib/types";

import { AGENT_CAPS, runEvents } from "./fixtures";

function build(events: (e: ReturnType<typeof runEvents>) => RunEvent[], now = 0): LiveRun {
  const e = runEvents();
  return events(e).reduce((run, ev) => applyEvent(run, ev, now), newRun(now));
}

const started = (e: ReturnType<typeof runEvents>) => [
  e("run_started", { capabilities: AGENT_CAPS }),
  e("step_started", { step_id: "s1", kind: "request", label: "Request received", status: "running" }),
  e("step_completed", { step_id: "s1", kind: "request", label: "Request received", status: "completed", detail: "Tenant scope from the verified principal", duration_ms: 0.4 }),
  e("step_started", { step_id: "s2", kind: "model", label: "Agent orchestration", status: "running", detail: "Deciding the next step", metadata: { call: 1, provider: "gemini" } }),
];

const row = (kind: string) => screen.getAllByTestId("trace-step").find((r) => r.dataset.kind === kind)!;

function mockReducedMotion(reduced: boolean) {
  window.matchMedia = vi.fn().mockImplementation((q: string) => ({
    matches: reduced && q.includes("reduce"),
    media: q,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

afterEach(() => {
  vi.useRealTimers();
  // @ts-expect-error reset the jsdom stub
  delete window.matchMedia;
});

describe("LiveTrace", () => {
  it("before any backend event: transport state only, no invented agent steps", () => {
    render(<LiveTrace run={newRun(0)} />);
    expect(screen.getAllByTestId("transport-step").map((r) => r.textContent)).toEqual(["Sending request… — in progress", "Waiting for agent events"]);
    expect(screen.queryAllByTestId("trace-step")).toHaveLength(0);
    expect(screen.getByTestId("run-summary")).toHaveTextContent("Sending request…");
  });

  it("completed request is green, the running step is amber, capabilities are gray", () => {
    render(<LiveTrace run={build(started)} />);
    const request = row("request");
    expect(request.dataset.status).toBe("completed");
    expect(within(request).getByText(/Completed/)).toBeInTheDocument();
    expect(request.querySelector(".bg-emerald-50")).not.toBeNull();
    const model = row("model");
    expect(model.dataset.status).toBe("running");
    expect(model).toHaveClass("bg-amber-50/70");
    expect(within(model).getByTestId("running-dot")).toHaveClass("bg-amber-500");
    expect(within(model).getByText(/Running/)).toBeInTheDocument();
    expect(within(model).getByText("LangGraph")).toBeInTheDocument();
    expect(within(model).getByText("Gemini")).toBeInTheDocument();
    expect(screen.getAllByTestId("trace-placeholder").map((p) => p.dataset.kind)).toEqual([
      "commerce_tool",
      "retrieval",
      "grounding",
      "action_proposal",
      "response",
    ]);
    expect(screen.getAllByTestId("trace-placeholder")[0]).toHaveTextContent("Not started");
  });

  it("the active step animates; reduced motion removes every animation but keeps status", () => {
    mockReducedMotion(false);
    const { unmount } = render(<LiveTrace run={build(started)} />);
    expect(row("model")).toHaveClass("trace-row-running");
    expect(row("model")).toHaveAttribute("aria-current", "step");
    expect(within(row("model")).getByTestId("running-dot")).toHaveClass("trace-dot-running");
    unmount();
    mockReducedMotion(true);
    render(<LiveTrace run={build(started)} />);
    expect(row("model")).not.toHaveClass("trace-row-running");
    expect(row("model").dataset.motion).toBe("reduced");
    expect(within(row("model")).getByTestId("running-dot")).not.toHaveClass("trace-dot-running");
    expect(row("model").dataset.status).toBe("running");
    expect(within(row("model")).getByText(/Running/)).toBeInTheDocument();
  });

  it("shows a live elapsed timer while running; the backend duration replaces it", () => {
    vi.useFakeTimers();
    vi.setSystemTime(10_000);
    const run = build(started, 10_000);
    const { rerender } = render(<LiveTrace run={run} />);
    act(() => {
      vi.advanceTimersByTime(1300);
    });
    expect(within(row("model")).getByTestId("elapsed")).toHaveTextContent("1.3s");
    expect(screen.getByTestId("run-summary")).toHaveTextContent(/^Running · 1\.3s$/);
    const done = applyEvent(run, runEvents()("step_completed", { run_id: run.runId!, sequence: 99, step_id: "s2", kind: "model", label: "Agent orchestration", status: "completed", duration_ms: 812, metadata: { call: 1, provider: "gemini" } }), 11_300);
    rerender(<LiveTrace run={done} />);
    expect(within(row("model")).queryByTestId("elapsed")).toBeNull();
    expect(within(row("model")).getByTestId("duration")).toHaveTextContent("812ms");
    expect(row("model").dataset.status).toBe("completed");
  });

  it("a failed run: the failed step is red and announced, later steps skipped and muted", () => {
    const run = build((e) => [
      ...started(e),
      e("step_completed", { step_id: "s2", kind: "model", label: "Agent orchestration", status: "completed" }),
      e("step_started", { step_id: "s3", kind: "retrieval", label: "Policy retrieval", status: "running" }),
      e("step_failed", { step_id: "s3", kind: "retrieval", label: "Policy retrieval", status: "failed", detail: "Policy retrieval service was unavailable", metadata: { outcome: "error" } }),
      e("step_skipped", { step_id: "s4", kind: "grounding", label: "Grounding validation", status: "skipped", detail: "Not run: the run stopped" }),
      e("step_skipped", { step_id: "s5", kind: "response", label: "Response generated", status: "skipped", detail: "Not run: the run stopped" }),
      e("run_failed", { error: { code: "agent_retrieval_error", message: "Policy knowledge could not be retrieved.", status: 503 } }),
    ]);
    render(<LiveTrace run={run} />);
    const failed = row("retrieval");
    expect(failed.dataset.status).toBe("failed");
    expect(failed).toHaveAttribute("role", "alert");
    expect(failed).toHaveTextContent("Policy retrieval service was unavailable");
    expect(within(failed).getByText(/Failed/)).toBeInTheDocument();
    expect(failed.querySelector(".bg-rose-50")).not.toBeNull();
    for (const kind of ["grounding", "response"]) {
      expect(row(kind).dataset.status).toBe("skipped");
      expect(row(kind)).toHaveTextContent("Not run");
      expect(row(kind)).toHaveClass("opacity-70");
    }
    expect(screen.queryAllByTestId("trace-placeholder")).toHaveLength(0);
    expect(screen.queryAllByTestId("trace-step").some((r) => r.dataset.status === "completed" && r.dataset.kind === "response")).toBe(false);
    expect(screen.getByTestId("run-summary")).toHaveTextContent(/^Failed after/);
    expect(screen.getByTestId("run-error")).toHaveTextContent("Policy knowledge could not be retrieved.");
  });

  it("approval: a stable amber waiting state (no spinner), then the fixed next steps in gray", () => {
    let run = build((e) => [
      ...started(e),
      e("step_completed", { step_id: "s2", kind: "model", label: "Agent orchestration", status: "completed" }),
      e("step_completed", { step_id: "s3", kind: "action_proposal", label: "Action proposal", status: "completed" }),
      e("approval_required", {
        step_id: "s4",
        kind: "approval",
        label: "Human approval required",
        status: "waiting",
        detail: "Nothing changes until an approver decides",
        next_steps: [
          { kind: "action_execution", label: "Deterministic execution" },
          { kind: "response", label: "Response generated" },
        ],
      }),
    ]);
    run = { ...run, phase: "waiting_approval" };
    render(<LiveTrace run={run} />);
    const approval = row("approval");
    expect(approval.dataset.status).toBe("waiting");
    expect(approval).toHaveTextContent("HUMAN APPROVAL REQUIRED");
    expect(approval).toHaveTextContent("Waiting for approval");
    expect(within(approval).queryByTestId("running-dot")).toBeNull();
    expect(approval).not.toHaveClass("trace-row-running");
    expect(screen.getAllByTestId("trace-placeholder").map((p) => p.dataset.kind)).toEqual(["action_execution", "response"]);
    expect(screen.getByTestId("run-summary")).toHaveTextContent("Waiting for human approval");
  });
});
