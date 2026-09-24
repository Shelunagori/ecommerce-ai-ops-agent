import { describe, expect, it } from "vitest";

import { formatEffective, segmentAnswer } from "@/lib/citations";

import { V2, citation } from "./fixtures";

describe("segmentAnswer", () => {
  it("turns validated citations into numbered segments", () => {
    const s = segmentAnswer(`15% credit [${V2}], capped [${V2}].`, [citation]);
    const cites = s.filter((x) => x.kind === "citation");
    expect(cites).toHaveLength(2);
    expect(cites.every((c) => c.kind === "citation" && c.index === 1)).toBe(true);
    expect(s[0]).toEqual({ kind: "text", text: "15% credit " });
  });

  it("keeps unknown citations as plain text (never invents metadata)", () => {
    const s = segmentAnswer("See [policy://refund-policy/v9#chunk-1].", [citation]);
    expect(s.every((x) => x.kind === "text")).toBe(true);
  });

  it("handles answers without citations", () => {
    expect(segmentAnswer("Order ORD-1001 was delivered.", [])).toEqual([
      { kind: "text", text: "Order ORD-1001 was delivered." },
    ]);
  });

  it("formats effective ranges", () => {
    expect(formatEffective(citation)).toBe("since 2026-08-15");
    expect(formatEffective({ effective_from: "2026-03-01", effective_to: "2026-08-15" })).toBe("2026-03-01 – 2026-08-15");
  });
});
