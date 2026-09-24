/**
 * Single source of truth for talking to the CommerceOps API.
 *
 * NEXT_PUBLIC_API_URL is inlined at build time, so on Vercel it must be set
 * in the project's environment variables before building.
 */

const DEV_FALLBACK_URL = "http://localhost:8000";

function resolveApiUrl(): string | null {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configured) return configured.replace(/\/+$/, "");
  // Only fall back to localhost during `next dev`; never in a production build.
  return process.env.NODE_ENV === "development" ? DEV_FALLBACK_URL : null;
}

export const API_URL = resolveApiUrl();

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
  try {
    const res = await fetch(`${API_URL}${path}`, {
      headers: { Accept: "application/json", ...headers },
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    const data = (await res.json().catch(() => null)) as T | null;
    if (res.ok && data !== null) return { ok: true, status: res.status, data };
    const code = (data as { error?: { code?: string } } | null)?.error?.code;
    return { ok: false, status: res.status, data, error: code ?? `HTTP ${res.status}` };
  } catch (err) {
    const error = err instanceof Error && err.name === "TimeoutError" ? "Timed out" : "Unreachable";
    return { ok: false, status: null, data: null, error };
  }
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

/** JSON request with a stable error shape ({code, message, status}); never throws. */
export async function apiRequest<T>(
  path: string,
  {
    method = "GET",
    body,
    headers = {},
    timeoutMs = 90_000,
  }: { method?: string; body?: unknown; headers?: Record<string, string>; timeoutMs?: number } = {},
): Promise<{ ok: true; data: T } | { ok: false; error: { code: string; message: string; status: number | null } }> {
  if (!API_URL) {
    return { ok: false, error: { code: "api_not_configured", message: "NEXT_PUBLIC_API_URL is not set.", status: null } };
  }
  try {
    const res = await fetch(`${API_URL}${path}`, {
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
    if (res.ok && data !== null) return { ok: true, data };
    return {
      ok: false,
      error: {
        code: data?.error?.code ?? `http_${res.status}`,
        message: data?.error?.message ?? `Request failed (HTTP ${res.status}).`,
        status: res.status,
      },
    };
  } catch (err) {
    const timeout = err instanceof Error && err.name === "TimeoutError";
    return {
      ok: false,
      error: {
        code: timeout ? "timeout" : "unreachable",
        message: timeout ? "The request timed out." : "The API is unreachable.",
        status: null,
      },
    };
  }
}
