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

export async function apiGet<T>(path: string, timeoutMs = 5000): Promise<ApiResult<T>> {
  if (!API_URL) {
    return { ok: false, status: null, data: null, error: "NEXT_PUBLIC_API_URL is not set" };
  }
  try {
    const res = await fetch(`${API_URL}${path}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    const data = (await res.json().catch(() => null)) as T | null;
    if (res.ok && data !== null) return { ok: true, status: res.status, data };
    return { ok: false, status: res.status, data, error: `HTTP ${res.status}` };
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
