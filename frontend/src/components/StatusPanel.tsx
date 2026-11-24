"use client";

import { useCallback, useEffect, useState } from "react";

import { API_URL, getDatabaseHealth, getHealth } from "@/lib/api";
import { retryWake, useBackendWake } from "@/lib/backendWake";

type State = "checking" | "ok" | "down";
type Check = { state: State; detail: string };

const INITIAL: Check = { state: "checking", detail: "Checking…" };

const BADGE: Record<State, string> = {
  checking: "bg-slate-100 text-slate-600 ring-slate-200",
  ok: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  down: "bg-rose-50 text-rose-700 ring-rose-200",
};

const LABEL: Record<State, string> = { checking: "Checking", ok: "Online", down: "Offline" };

function StatusRow({ name, check }: { name: string; check: Check }) {
  // Render free tier: while a check waits on a sleeping API, say so instead of "Checking".
  const wake = useBackendWake().status;
  const waking = check.state === "checking" && wake === "waking";
  const unreachable = check.state === "checking" && wake === "unreachable";
  const detail = waking ? "Waking backend… please wait" : unreachable ? "Backend unreachable" : check.detail;
  const badge = waking
    ? "bg-orange-50 text-orange-700 ring-orange-200"
    : unreachable
      ? BADGE.down
      : BADGE[check.state];
  const label = waking ? "Waking" : unreachable ? "Unreachable" : LABEL[check.state];
  return (
    <li className="flex items-center justify-between gap-4 py-3">
      <div>
        <p className="font-medium text-slate-900">{name}</p>
        <p className="text-sm text-slate-500">{detail}</p>
      </div>
      <div className="flex items-center gap-2">
        {unreachable && (
          <button
            type="button"
            onClick={retryWake}
            className="rounded-md border border-rose-300 px-2 py-0.5 text-xs font-medium text-rose-700 hover:bg-rose-50"
          >
            Retry
          </button>
        )}
        <span
          className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${badge}`}
          data-testid={`${name.toLowerCase()}-status`}
        >
          {label}
        </span>
      </div>
    </li>
  );
}

export default function StatusPanel() {
  const [api, setApi] = useState<Check>(INITIAL);
  const [db, setDb] = useState<Check>(INITIAL);

  const refresh = useCallback(async () => {
    setApi(INITIAL);
    setDb(INITIAL);
    const [health, dbHealth] = await Promise.all([getHealth(), getDatabaseHealth()]);

    setApi(
      health.ok
        ? { state: "ok", detail: `${health.data.service} is responding` }
        : { state: "down", detail: health.error },
    );

    if (dbHealth.ok) {
      setDb({ state: "ok", detail: "PostgreSQL is reachable" });
    } else if (dbHealth.data) {
      setDb({ state: "down", detail: `Database ${dbHealth.data.database.replace("_", " ")}` });
    } else {
      setDb({ state: "down", detail: "Status unknown (API unreachable)" });
    }
  }, []);

  useEffect(() => {
    // Initial status fetch on mount; state updates happen after the awaited requests.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">System status</h2>
        <button
          type="button"
          onClick={() => void refresh()}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
        >
          Refresh
        </button>
      </div>
      <ul className="mt-2 divide-y divide-slate-100">
        <StatusRow name="API" check={api} />
        <StatusRow name="Database" check={db} />
      </ul>
      <p className="mt-4 text-xs text-slate-400">
        API endpoint: <code>{API_URL ?? "not configured"}</code>
      </p>
    </section>
  );
}
