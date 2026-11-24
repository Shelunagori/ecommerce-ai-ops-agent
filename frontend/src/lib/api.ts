/**
 * Single source of truth for talking to the CommerceOps API.
 *
 * NEXT_PUBLIC_API_URL is inlined at build time, so on Vercel it must be set
 * in the project's environment variables before building.
 */

import { reportBackendResponded, reportWakeSignal, SLOW_AFTER_MS, waitUntilAwake } from "./backendWake";

const DEV_FALLBACK_URL = "http://localhost:8000";

function resolveApiUrl(): string | null {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configured) return configured.replace(/\/+$/, "");
  // Only fall back to localhost during `next dev`; never in a production build.
  return process.env.NODE_ENV === "development" ? DEV_FALLBACK_URL : null;
}

export const API_URL = resolveApiUrl();

/** Render's edge answers these (without our JSON error body) while the free-tier API starts. */
const PROXY_WAKE_STATUSES = new Set([502, 503, 504]);

/** A failure that looks like a sleeping server rather than an application error. */
function looksAsleep(status: number | null, appErrorCode: string | undefined): boolean {
  return status === null || (PROXY_WAKE_STATUSES.has(status) && !appErrorCode);
}

type Attempt<R> = { result: R; asleep: boolean };

/**
 * Runs one request and reports to the wake-up tracker (src/lib/backendWake.ts). A read that
 * fails like a sleeping server waits for the wake-up and is retried ONCE; writes are never
 * retried, so an action can never be sent twice.
 */
async function withWake<R>(isRead: boolean, attempt: () => Promise<Attempt<R>>): Promise<R> {
  const slow = isRead ? setTimeout(reportWakeSignal, SLOW_AFTER_MS) : undefined;
  let first: Attempt<R>;
  try {
    first = await attempt();
  } finally {
    clearTimeout(slow);
  }
  if (!first.asleep) {
    reportBackendResponded();
    return first.result;
  }
  reportWakeSignal();
  if (!isRead || !(await waitUntilAwake())) return first.result;
  return (await attempt()).result;
}

export type ApiResult<T> =
  | { ok: true; status: number; data: T }
  | { ok: false; status: number | null; data: T | null; error: string };

export async function apiGet<T>(
  path: string,
  { headers = {}, timeoutMs = 5000 }: { headers?: Record<string, string>; timeoutMs?: number } = {},
): Promise<ApiResult<T>> {
  if (!API_URL) {
    return { ok: false, status: null, data: null, error: "NEXT_PUBLIC_API_URL is not set" };
  }
  const url = `${API_URL}${path}`;
  return withWake(true, async (): Promise<Attempt<ApiResult<T>>> => {
    try {
      const res = await fetch(url, {
        headers: { Accept: "application/json", ...headers },
        cache: "no-store",
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = (await res.json().catch(() => null)) as T | null;
      if (res.ok && data !== null) return { result: { ok: true, status: res.status, data }, asleep: false };
      const code = (data as { error?: { code?: string } } | null)?.error?.code;
      return {
        result: { ok: false, status: res.status, data, error: code ?? `HTTP ${res.status}` },
        asleep: looksAsleep(res.status, code),
      };
    } catch (err) {
      const error = err instanceof Error && err.name === "TimeoutError" ? "Timed out" : "Unreachable";
      return { result: { ok: false, status: null, data: null, error }, asleep: true };
    }
  });
}

export type HealthResponse = { status: "ok"; service: string };
export type DatabaseHealthResponse = {
  status: "ok" | "error";
  database: "reachable" | "unreachable" | "not_configured";
};

export const getHealth = () => apiGet<HealthResponse>("/health");
export const getDatabaseHealth = () => apiGet<DatabaseHealthResponse>("/health/db");

export type DemoSummary = {
  tenant: { name: string; slug: string };
  customers: number;
  products: number;
  orders: number;
  unpaid_invoices: number;
  overdue_invoices: number;
  delayed_shipments: number;
};

/** Demo tenant context header (NOT authentication; see src/lib/demo.ts). */
export const getDemoSummary = (tenantId: string) =>
  apiGet<DemoSummary>("/api/demo/summary", { headers: { "X-Tenant-ID": tenantId } });

export type ApiResponse<T> =
  | { ok: true; data: T }
  | { ok: false; error: { code: string; message: string; status: number | null } };

/** JSON request with a stable error shape ({code, message, status}); never throws. */
export async function apiRequest<T>(
  path: string,
  {
    method = "GET",
    body,
    headers = {},
    timeoutMs = 90_000,
  }: { method?: string; body?: unknown; headers?: Record<string, string>; timeoutMs?: number } = {},
): Promise<ApiResponse<T>> {
  if (!API_URL) {
    return { ok: false, error: { code: "api_not_configured", message: "NEXT_PUBLIC_API_URL is not set.", status: null } };
  }
  const url = `${API_URL}${path}`;
  return withWake(method.toUpperCase() === "GET", async (): Promise<Attempt<ApiResponse<T>>> => {
    try {
      const res = await fetch(url, {
        method,
        headers: {
          Accept: "application/json",
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          ...headers,
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = (await res.json().catch(() => null)) as
        | (T & { error?: { code?: string; message?: string } })
        | null;
      if (res.ok && data !== null) return { result: { ok: true, data }, asleep: false };
      return {
        result: {
          ok: false,
          error: {
            code: data?.error?.code ?? `http_${res.status}`,
            message: data?.error?.message ?? `Request failed (HTTP ${res.status}).`,
            status: res.status,
          },
        },
        asleep: looksAsleep(res.status, data?.error?.code),
      };
    } catch (err) {
      const timeout = err instanceof Error && err.name === "TimeoutError";
      return {
        result: {
          ok: false,
          error: {
            code: timeout ? "timeout" : "unreachable",
            message: timeout ? "The request timed out." : "The API is unreachable.",
            status: null,
          },
        },
        asleep: true,
      };
    }
  });
}
