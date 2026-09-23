"use client";

import { useCallback, useEffect, useState } from "react";

import { type DemoSummary, getDemoSummary } from "@/lib/api";
import { DEFAULT_DEMO_TENANT, DEMO_TENANTS } from "@/lib/demo";

const METRICS: { key: keyof Omit<DemoSummary, "tenant">; label: string }[] = [
  { key: "customers", label: "Customers" },
  { key: "orders", label: "Orders" },
  { key: "overdue_invoices", label: "Overdue invoices" },
  { key: "delayed_shipments", label: "Delayed shipments" },
];

type State =
  | { kind: "loading" }
  | { kind: "ok"; summary: DemoSummary }
  | { kind: "error"; message: string };

export default function DemoDataPanel() {
  const [tenantId, setTenantId] = useState(DEFAULT_DEMO_TENANT.id);
  const [state, setState] = useState<State>({ kind: "loading" });

  const load = useCallback(async (id: string) => {
    setState({ kind: "loading" });
    const res = await getDemoSummary(id);
    setState(
      res.ok
        ? { kind: "ok", summary: res.data }
        : {
            kind: "error",
            message:
              res.error === "tenant_not_found"
                ? "Demo tenant not found — run the seed script."
                : res.error,
          },
    );
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load(tenantId);
  }, [load, tenantId]);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-lg font-semibold text-slate-900">Demo data</h2>
        <select
          aria-label="Demo tenant"
          value={tenantId}
          onChange={(e) => setTenantId(e.target.value)}
          className="rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700"
        >
          {DEMO_TENANTS.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>

      {state.kind === "error" ? (
        <p className="mt-4 text-sm text-rose-700" data-testid="demo-error">
          {state.message}
        </p>
      ) : (
        <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {METRICS.map(({ key, label }) => (
            <div key={key} className="rounded-lg bg-slate-50 p-3">
              <dt className="text-xs text-slate-500">{label}</dt>
              <dd className="mt-1 text-2xl font-semibold text-slate-900" data-testid={`demo-${key}`}>
                {state.kind === "ok" ? state.summary[key] : "…"}
              </dd>
            </div>
          ))}
        </dl>
      )}

      <p className="mt-4 text-xs text-slate-400">
        Synthetic data. Tenant is selected with the <code>X-Tenant-ID</code> demo header — not
        authentication.
      </p>
    </section>
  );
}
