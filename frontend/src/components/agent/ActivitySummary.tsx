import type { RetrievalSummary, ToolCallSummary } from "@/lib/types";

const TOOL_LABEL: Record<string, string> = {
  get_customer: "Customer lookup",
  search_customers: "Customer search",
  get_order: "Order lookup",
  list_customer_orders: "Customer orders",
  get_latest_customer_order: "Latest order",
  get_invoice: "Invoice lookup",
  get_latest_unpaid_invoice: "Unpaid invoice",
  get_shipment: "Shipment lookup",
  get_order_shipments: "Order shipments",
  list_delayed_shipments: "Delayed shipments",
  get_product: "Product lookup",
  search_products: "Product search",
};

/** Safe summary of what the assistant used: names, outcomes, counts - never raw payloads. */
export default function ActivitySummary({
  toolCalls,
  retrievals,
}: {
  toolCalls: ToolCallSummary[];
  retrievals: RetrievalSummary[];
}) {
  if (toolCalls.length === 0 && retrievals.length === 0) return null;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Assistant activity" data-testid="activity">
      {toolCalls.map((c, i) => (
        <li
          key={`t${i}`}
          className={`rounded-md px-2 py-0.5 text-xs ring-1 ring-inset ${
            c.outcome === "success" ? "bg-slate-50 text-slate-700 ring-slate-200" : "bg-rose-50 text-rose-700 ring-rose-200"
          }`}
        >
          {TOOL_LABEL[c.tool] ?? c.tool}
          {c.outcome !== "success" && ` · ${c.outcome.replaceAll("_", " ")}`}
        </li>
      ))}
      {retrievals.map((r, i) => (
        <li key={`r${i}`} className="rounded-md bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700 ring-1 ring-inset ring-indigo-200">
          Policy search ·{" "}
          {r.outcome === "success"
            ? `${r.result_count} source${r.result_count === 1 ? "" : "s"}${r.as_of ? ` as of ${r.as_of}` : ""}`
            : r.outcome.replaceAll("_", " ")}
        </li>
      ))}
    </ul>
  );
}
