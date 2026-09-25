import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ExecutionTrace from "@/components/agent/trace/ExecutionTrace";

import { APPROVAL_TRACE, RAG_TRACE, TOOLS_TRACE, ev, response } from "./fixtures";

const steps = () => screen.getAllByTestId("trace-step");

describe("ExecutionTrace", () => {
  it("renders a tools-only run in the backend's order with commerce badges", () => {
    render(<ExecutionTrace response={response({ execution_trace: TOOLS_TRACE, duration_ms: 1500, tool_calls: [] })} />);
    expect(steps().map((s) => s.dataset.kind)).toEqual(["request", "model", "commerce_tool", "model", "response"]);
    const tool = steps()[2];
    expect(tool).toHaveTextContent("Tool: get_shipment");
    expect(within(tool).getByText("LangChain Tool")).toBeInTheDocument();
    expect(within(tool).getByText("PostgreSQL")).toBeInTheDocument();
    expect(within(steps()[1]).getByText("Gemini")).toBeInTheDocument();
    expect(screen.queryByText("pgvector")).not.toBeInTheDocument(); // no RAG claimed
    expect(screen.getByTestId("run-summary")).toHaveTextContent("1.5s · 2 model calls · 0 tools");
  });

  it("renders a RAG-only run with pgvector and grounding", () => {
    render(
      <ExecutionTrace
        response={response({
          execution_trace: RAG_TRACE,
          duration_ms: 17500,
          retrievals: [{ round: 1, as_of: "2026-09-01", result_count: 3, citations: [], outcome: "success", error_code: null, rejected_argument_names: [], duration_ms: 20 }],
        })}
      />,
    );
    const retrieval = steps().find((s) => s.dataset.kind === "retrieval")!;
    expect(within(retrieval).getByText("pgvector")).toBeInTheDocument();
    expect(within(retrieval).getByText("RAG")).toBeInTheDocument();
    expect(retrieval).toHaveTextContent("3 eligible policy sections");
    expect(screen.getByText("Grounding validation")).toBeInTheDocument();
    expect(screen.getByTestId("run-summary")).toHaveTextContent("17.5s · 2 model calls · 0 tools · 3 retrieved sources");
  });

  it("labels lexical retrieval honestly (no pgvector badge)", () => {
    const trace = [ev(1, "retrieval", "Policy retrieval", { retriever: "lexical-pg-fts-v1" }), ev(2, "response", "Response generated")];
    render(<ExecutionTrace response={response({ execution_trace: trace })} />);
    expect(screen.getByText("PostgreSQL full-text")).toBeInTheDocument();
    expect(screen.queryByText("pgvector")).not.toBeInTheDocument();
  });

  it("renders a mixed run exactly in the given order", () => {
    const mixed = [
      ...TOOLS_TRACE.slice(0, 3),
      ev(4, "model", "Agent orchestration", { call: 2 }),
      ev(5, "retrieval", "Policy retrieval", { retriever: "semantic-pgvector-v1" }),
      ev(6, "model", "Agent orchestration", { call: 3 }),
      ev(7, "grounding", "Grounding validation"),
      ev(8, "response", "Response generated"),
    ];
    render(<ExecutionTrace response={response({ execution_trace: mixed })} />);
    expect(steps().map((s) => s.dataset.kind)).toEqual([
      "request", "model", "commerce_tool", "model", "retrieval", "model", "grounding", "response",
    ]);
  });

  it("distinguishes waiting approval and failed steps", () => {
    const trace = [...APPROVAL_TRACE.slice(0, 4), ev(5, "commerce_tool", "Tool: x", { outcome: "unknown_tool" }, "failed")];
    render(<ExecutionTrace response={response({ execution_trace: trace })} />);
    const approval = steps().find((s) => s.dataset.kind === "approval")!;
    expect(approval.dataset.status).toBe("waiting");
    expect(approval).toHaveTextContent("HUMAN APPROVAL REQUIRED");
    expect(within(approval).getByText("HITL")).toBeInTheDocument();
    expect(within(approval).getByRole("img", { name: "Waiting" })).toBeInTheDocument();
    const failed = steps().find((s) => s.dataset.status === "failed")!;
    expect(within(failed).getByRole("img", { name: "Failed" })).toBeInTheDocument();
  });

  it("says when no trace exists (older history)", () => {
    render(<ExecutionTrace response={undefined} />);
    expect(screen.getByTestId("trace-unavailable")).toHaveTextContent("Execution trace is available for new runs.");
  });
});
