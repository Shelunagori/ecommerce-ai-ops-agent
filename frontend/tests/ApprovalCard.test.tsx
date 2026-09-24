import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ApprovalCard from "@/components/agent/ApprovalCard";
import { toView } from "@/components/agent/ChatPanel";

import { pendingAction } from "./fixtures";

const view = toView(pendingAction);

describe("ApprovalCard", () => {
  it("shows the exact arguments and lets an approver approve", async () => {
    const onDecide = vi.fn().mockResolvedValue(null);
    render(<ApprovalCard action={view} canApprove onDecide={onDecide} />);
    expect(screen.getByTestId("action-status")).toHaveTextContent("Pending approval");
    expect(screen.getByText("ORD-1004")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(onDecide).toHaveBeenCalledWith("approve");
  });

  it("reject calls the reject decision", async () => {
    const onDecide = vi.fn().mockResolvedValue(null);
    render(<ApprovalCard action={view} canApprove onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(onDecide).toHaveBeenCalledWith("reject");
  });

  it("disables decisions for non-approvers", () => {
    render(<ApprovalCard action={view} canApprove={false} onDecide={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(screen.getByText(/Only approvers/)).toBeInTheDocument();
  });

  it.each([
    ["succeeded", "Succeeded"],
    ["rejected", "Rejected"],
    ["expired", "Expired"],
    ["executing", "Executing"],
    ["approved", "Approved"],
  ] as const)("resolved status %s has no decision buttons", (status, label) => {
    render(<ApprovalCard action={{ ...view, status }} canApprove onDecide={vi.fn()} />);
    expect(screen.getByTestId("action-status")).toHaveTextContent(label);
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
  });

  it("shows failures and decision errors", async () => {
    const { rerender } = render(
      <ApprovalCard action={{ ...view, status: "failed", failure_code: "order_not_cancellable" }} canApprove onDecide={vi.fn()} />,
    );
    expect(screen.getByText(/order_not_cancellable/)).toBeInTheDocument();
    const onDecide = vi.fn().mockResolvedValue("The approval window for this action has expired.");
    rerender(<ApprovalCard action={view} canApprove onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("expired");
  });
});
