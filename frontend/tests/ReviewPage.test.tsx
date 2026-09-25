import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import ReviewPage from "@/app/review/page";
import { REPO_URL } from "@/components/review/content";

describe("/review (public, no auth)", () => {
  it("renders the case study with navigation, CTAs and architecture", () => {
    render(<ReviewPage />);
    expect(screen.getByRole("heading", { level: 1, name: "CommerceOps AI" })).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "Review sections" });
    for (const label of ["Overview", "Architecture", "Agent Flow", "RAG", "HITL", "Security", "Evaluation", "Deployment"]) {
      expect(within(nav).getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(screen.getAllByRole("link", { name: /Try Live Demo|Launch Live Demo/ })[0]).toHaveAttribute("href", "/?demo=1");
    expect(screen.getByRole("link", { name: "Open Application" })).toHaveAttribute("href", "/");
    expect(screen.getAllByRole("link", { name: "View GitHub" })[0]).toHaveAttribute("href", REPO_URL);
    const diagram = screen.getByTestId("architecture-diagram");
    for (const text of ["LangGraph orchestration", "Supabase Auth", "LLM provider layer", "Approval-gated actions", "LangGraph checkpoints"]) {
      expect(within(diagram).getByText(text)).toBeInTheDocument();
    }
  });

  it("walkthrough tabs switch flows and the HITL flow shows the write boundary", async () => {
    render(<ReviewPage />);
    const flows = screen.getByTestId("flow-walkthrough");
    expect(within(flows).getByRole("tab", { name: "Commerce query" })).toHaveAttribute("aria-selected", "true");
    expect(within(flows).getByText("get_shipment")).toBeInTheDocument();
    await userEvent.click(within(flows).getByRole("tab", { name: "Mixed query" }));
    expect(within(flows).getByText(/what compensation applies if it is delayed/)).toBeInTheDocument();
    await userEvent.click(within(flows).getByRole("tab", { name: "HITL action" }));
    expect(within(flows).getByText("HUMAN APPROVAL REQUIRED")).toBeInTheDocument();
    expect(within(flows).getByText("Write boundary")).toBeInTheDocument();
  });

  it("shows only filled-in evidence and no credentials", () => {
    render(<ReviewPage />);
    const text = document.body.textContent ?? "";
    expect(text).not.toContain("__"); // no unfilled placeholders
    expect(text).not.toMatch(/service_role|anon key|password|postgresql:\/\/|GEMINI_API_KEY=/i);
    expect(screen.getByText("hit@1 1.00 vs 0.40")).toBeInTheDocument();
  });
  it("explains the live trace and labels its illustration as not a live run", () => {
    render(<ReviewPage />);
    const section = screen.getByRole("region", { name: "Real-time execution trace" });
    expect(within(section).getByText("Not chain-of-thought")).toBeInTheDocument();
    expect(within(section).getByText(/never frontend timers/)).toBeInTheDocument();
    const figure = within(section).getByTestId("live-trace-illustration");
    expect(within(figure).getByText("UI illustration — not a live run")).toBeInTheDocument();
    expect(within(screen.getByRole("navigation", { name: "Review sections" })).getByRole("link", { name: "Live Trace" })).toHaveAttribute("href", "#live-trace");
  });
});
