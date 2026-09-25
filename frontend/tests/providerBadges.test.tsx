import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ChatPanel from "@/components/agent/ChatPanel";
import { badgesFor } from "@/components/agent/trace/TechnologyBadge";
import type { Membership } from "@/lib/types";

import { NS, mockFetch } from "./fixtures";
import userEvent from "@testing-library/user-event";

describe("model provider badges", () => {
  it("names the provider that actually answered each model call", () => {
    expect(badgesFor({ kind: "model", metadata: { provider: "cloudflare", call: 1 } })).toEqual([
      "LangGraph",
      "Cloudflare Workers AI",
    ]);
    expect(badgesFor({ kind: "model", metadata: { provider: "gemini", call: 2, fallback_used: true } })).toEqual([
      "LangGraph",
      "Gemini",
      "Provider fallback",
    ]);
    expect(badgesFor({ kind: "model", metadata: { provider: "gemini" } })).toEqual(["LangGraph", "Gemini"]);
    expect(badgesFor({ kind: "model", metadata: { provider: "something-else" } })).toEqual(["LangGraph"]);
  });
});

describe("provider errors are shown as short safe messages", () => {
  const tenant: Membership = { tenant_id: NS, slug: "northstar-commerce", name: "Northstar Commerce", role: "member" };
  const cases: Array<[string, string]> = [
    ["llm_rate_limited", "The AI model is busy right now. Please try again in a minute."],
    ["llm_unavailable", "The AI model is temporarily unavailable. Please try again in a minute."],
    ["llm_timeout", "The AI model took too long to respond. Please try again."],
    ["public_demo_limit_reached", "Public demo limit reached. Please start a reviewer session or try again later."],
  ];
  it.each(cases)("%s", async (code, text) => {
    mockFetch([
      { path: "/api/agent/threads/t-1/messages", body: { thread_id: "t-1", messages: [], pending_action: null } },
      { method: "POST", path: "/api/agent/messages", status: code.startsWith("public") ? 429 : 503, body: { error: { code, message: "Cloudflare 429 raw body {errors:[...]}" } } },
    ]);
    render(<ChatPanel tenant={tenant} threadId="t-1" />);
    await userEvent.type(screen.getByLabelText("Message"), "hi");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(text);
    expect(within(alert).queryByText(/raw body/)).toBeNull();
  });
});
