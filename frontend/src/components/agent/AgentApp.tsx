"use client";

import { useCallback, useEffect, useState } from "react";

import { getMe, newThreadId } from "@/lib/agent";
import { AUTH_MODE, supabase } from "@/lib/auth";
import type { ApiError, Me, Membership } from "@/lib/types";

import ChatPanel from "./ChatPanel";
import SignIn from "./SignIn";

function Workspace({ onSignOut }: { onSignOut?: () => void }) {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [tenantId, setTenantId] = useState<string | null>(null);
  const [threads, setThreads] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    setError(null);
    const res = await getMe();
    if (!res.ok) {
      setError(res.error);
      return;
    }
    setMe(res.data);
    setTenantId((current) => current ?? res.data.memberships[0]?.tenant_id ?? null);
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial fetch on mount
    void load();
  }, [load]);

  const tenant: Membership | undefined = me?.memberships.find((m) => m.tenant_id === tenantId);
  const threadId = tenant ? (threads[tenant.tenant_id] ??= newThreadId()) : null;

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
        <div className="mr-auto">
          <p className="text-base font-bold tracking-tight text-slate-900">CommerceOps AI</p>
          <p className="text-xs text-slate-500">Operations assistant · synthetic data only</p>
        </div>
        {me && me.memberships.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <span>Tenant</span>
            <select
              aria-label="Tenant"
              value={tenantId ?? ""}
              onChange={(e) => setTenantId(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none"
            >
              {me.memberships.map((m) => (
                <option key={m.tenant_id} value={m.tenant_id}>
                  {m.name} ({m.role})
                </option>
              ))}
            </select>
          </label>
        )}
        {tenant && (
          <button
            type="button"
            onClick={() => setThreads((t) => ({ ...t, [tenant.tenant_id]: newThreadId() }))}
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50"
          >
            New conversation
          </button>
        )}
        {onSignOut && (
          <button type="button" onClick={onSignOut} className="text-sm font-medium text-slate-600 hover:text-slate-900">
            Sign out
          </button>
        )}
        {me?.auth_mode === "demo" && (
          <span className="w-full rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800 ring-1 ring-amber-200 sm:w-auto">
            Local demo mode — not authenticated
          </span>
        )}
      </header>
      {error && (
        <div role="alert" className="m-4 flex items-center gap-3 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-800 ring-1 ring-rose-200">
          <span className="flex-1">
            {error.code === "auth_required" || error.code === "auth_invalid"
              ? "Your session is not valid. Sign in again."
              : `Could not load your workspace: ${error.message}`}
          </span>
          <button type="button" onClick={() => void load()} className="font-medium underline">
            Retry
          </button>
        </div>
      )}
      {me && me.memberships.length === 0 && (
        <p className="m-6 text-sm text-slate-600">Your account has no tenant access yet. Ask an administrator to grant a membership.</p>
      )}
      <main className="mx-auto flex min-h-0 w-full max-w-4xl flex-1 flex-col">
        {tenant && threadId && <ChatPanel key={`${tenant.tenant_id}:${threadId}`} tenant={tenant} threadId={threadId} />}
      </main>
    </div>
  );
}

export default function AgentApp() {
  const [session, setSession] = useState<"unknown" | "in" | "out">(AUTH_MODE === "demo" ? "in" : "unknown");

  useEffect(() => {
    const sb = supabase();
    if (AUTH_MODE === "demo") return;
    if (!sb) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- no auth client configured
      setSession("out");
      return;
    }
    sb.auth.getSession().then(({ data }) => setSession(data.session ? "in" : "out"));
    const { data } = sb.auth.onAuthStateChange((_e, s) => setSession(s ? "in" : "out"));
    return () => data.subscription.unsubscribe();
  }, []);

  if (session === "unknown") return <p className="p-6 text-sm text-slate-500">Loading…</p>;
  if (session === "out") return <SignIn />;
  return <Workspace onSignOut={AUTH_MODE === "supabase" ? () => void supabase()?.auth.signOut() : undefined} />;
}
