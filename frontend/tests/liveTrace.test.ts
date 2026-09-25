import { describe, expect, it } from "vitest";

import { applyEvent, completeRun, newRun, placeholders, visibleSteps } from "@/lib/liveTrace";

import { AGENT_CAPS, TOOLS_TRACE, runEvents } from "./fixtures";

describe("live trace reducer", () => {
  it("only backend events create steps; placeholders are unseen capabilities", () => {
    const e = runEvents();
    let run = newRun(0);
    expect(placeholders(run)).toEqual([]);
    run = applyEvent(run, e("run_started", { capabilities: AGENT_CAPS }), 1);
    expect(run.phase).toBe("running");
    run = applyEvent(run, e("step_started", { step_id: "s1", kind: "commerce_tool", label: "Tool: get_order", status: "running" }), 2);
    expect(visibleSteps(run).map((s) => [s.kind, s.status])).toEqual([["commerce_tool", "running"]]);
    expect(placeholders(run).map((c) => c.kind)).toEqual(["retrieval", "grounding", "action_proposal", "response"]);
    run = applyEvent(run, e("step_completed", { step_id: "s1", kind: "commerce_tool", label: "Tool: get_order", status: "completed", duration_ms: 12 }), 3);
    expect(visibleSteps(run)[0]).toMatchObject({ status: "completed", durationMs: 12, startedAt: 2 });
  });

  it("ignores events from an older run", () => {
    const run = applyEvent(newRun(0), runEvents("a".repeat(32))("run_started", { capabilities: [] }), 1);
    const stale = runEvents("b".repeat(32))("step_started", { step_id: "s1", kind: "model", label: "x", status: "running" });
    expect(applyEvent(run, stale, 2)).toBe(run);
  });

  it("a gray capability becomes 'skipped' only when the backend says so", () => {
    const e = runEvents();
    let run = applyEvent(newRun(0), e("run_started", { capabilities: AGENT_CAPS }), 1);
    expect(visibleSteps(run).some((s) => s.status === "skipped")).toBe(false);
    run = applyEvent(run, e("step_skipped", { step_id: "s9", kind: "retrieval", label: "Policy retrieval", status: "skipped", detail: "Not used in this run" }), 2);
    expect(visibleSteps(run).map((s) => [s.kind, s.status])).toEqual([["retrieval", "skipped"]]);
    expect(placeholders(run).map((c) => c.kind)).not.toContain("retrieval");
  });

  it("converges on the final trace, keeping live-measured durations it lacks", () => {
    const e = runEvents();
    let run = applyEvent(newRun(0), e("run_started", { capabilities: AGENT_CAPS }), 1);
    const steps = [
      ["request", "Request received", 0.5],
      ["model", "Agent orchestration", 800],
      ["commerce_tool", "Tool: get_shipment", 4.2],
      ["model", "Agent orchestration", 600],
      ["response", "Response generated", null],
    ] as const;
    steps.forEach(([kind, label, ms], i) => {
      run = applyEvent(run, e("step_completed", { step_id: `s${i + 1}`, kind, label, status: "completed", duration_ms: ms ?? undefined }), 2);
    });
    run = applyEvent(run, e("step_skipped", { step_id: "s9", kind: "retrieval", label: "Policy retrieval", status: "skipped" }), 3);
    run = completeRun(run, TOOLS_TRACE, { durationMs: 1500, modelCalls: 2, tools: 1, retrievedSources: null, citations: 0, actionStatus: null }, 4, false);
    const rows = visibleSteps(run);
    expect(rows.map((r) => r.kind)).toEqual(["request", "model", "commerce_tool", "model", "response", "retrieval"]);
    expect(rows[1].durationMs).toBe(800); // model duration measured live
    expect(rows[2].durationMs).toBe(4.2); // tool duration from the final trace
    expect(rows[5].status).toBe("skipped");
    expect(placeholders(run)).toEqual([]);
  });
});
