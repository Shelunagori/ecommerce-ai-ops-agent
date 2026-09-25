"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { startPublicDemo, supabase } from "@/lib/auth";

function ReviewerForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sb = supabase();

  return (
    <form
      aria-label="Reviewer sign in"
      className="space-y-4 text-left"
      onSubmit={async (e) => {
        e.preventDefault();
        if (!sb) return;
        setBusy(true);
        setError(null);
        const { error: err } = await sb.auth.signInWithPassword({ email, password });
        setBusy(false);
        if (err) setError("Sign-in failed. Check your email and password.");
      }}
    >
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-slate-700">
          Email
        </label>
        <input
          id="email"
          type="email"
          required
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
        />
      </div>
      <div>
        <label htmlFor="password" className="block text-sm font-medium text-slate-700">
          Password
        </label>
        <input
          id="password"
          type="password"
          required
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
        />
      </div>
      {error && (
        <p role="alert" className="text-sm text-rose-700">
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy || !sb}
        className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:opacity-50"
      >
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}

/**
 * Unauthenticated landing: "Try Live Demo" (Supabase anonymous session, read-only demo
 * tenant decided by the backend), "Reviewer Sign In" (email/password, tenant memberships)
 * and a link to the public engineering review.
 */
export default function SignIn() {
  const [showReviewer, setShowReviewer] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const autoStarted = useRef(false);
  const configured = supabase() !== null;

  const tryDemo = useCallback(async () => {
    setStarting(true);
    setError(null);
    const res = await startPublicDemo();
    setStarting(false);
    // On success the auth state listener in AgentApp switches to the workspace.
    if (!res.ok) setError(res.message);
  }, []);

  useEffect(() => {
    // `/?demo=1` (the /review page's "Try Live Demo" link) starts the demo directly.
    if (autoStarted.current || !configured) return;
    if (new URLSearchParams(window.location.search).get("demo") === "1") {
      autoStarted.current = true;
      window.history.replaceState(null, "", window.location.pathname);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-shot start from the URL
      void tryDemo();
    }
  }, [configured, tryDemo]);

  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 text-center shadow-sm ring-1 ring-slate-200">
        <p className="text-xs font-semibold uppercase tracking-widest text-indigo-600">Portfolio system</p>
        <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">CommerceOps AI</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Production-grade AI operations agent for multi-tenant ecommerce workflows.
        </p>

        {!configured ? (
          <p role="alert" className="mt-6 rounded-lg bg-rose-50 p-3 text-sm text-rose-800">
            Sign-in is not configured for this deployment.
          </p>
        ) : (
          <div className="mt-6 space-y-3">
            <button
              type="button"
              onClick={() => void tryDemo()}
              disabled={starting}
              className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:opacity-60"
            >
              {starting ? "Starting demo…" : "Try Live Demo"}
            </button>
            <button
              type="button"
              aria-expanded={showReviewer}
              aria-controls="reviewer-sign-in"
              onClick={() => setShowReviewer((v) => !v)}
              className="w-full rounded-lg px-4 py-2.5 text-sm font-semibold text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              Reviewer Sign In
            </button>
            <div role="status" aria-live="polite" className="min-h-0">
              {error && (
                <div className="rounded-lg bg-rose-50 p-3 text-left text-sm text-rose-800 ring-1 ring-rose-200">
                  <p>{error}</p>
                  <button type="button" onClick={() => void tryDemo()} className="mt-1 font-medium underline">
                    Retry
                  </button>
                </div>
              )}
            </div>
            {showReviewer && (
              <div id="reviewer-sign-in" className="border-t border-slate-100 pt-4">
                <ReviewerForm />
              </div>
            )}
          </div>
        )}

        <Link
          href="/review"
          className="mt-6 inline-block text-sm font-medium text-indigo-700 hover:text-indigo-900 focus:outline-none focus-visible:underline"
        >
          Review the Engineering →
        </Link>
        <p className="mt-4 text-xs text-slate-500">Synthetic data only.</p>
      </div>
    </main>
  );
}
