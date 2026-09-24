/**
 * Browser-side identity.
 *
 * NEXT_PUBLIC_AUTH_MODE=demo (default, local only): no login; the tenant selector sends
 * X-Tenant-ID, which the backend trusts ONLY outside production.
 * NEXT_PUBLIC_AUTH_MODE=supabase: Supabase Auth in the browser (public URL + anon /
 * publishable key only - never a service key); every API call carries the user's access
 * token, and the backend maps the verified user to allowed tenants.
 */
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

export type AuthMode = "demo" | "supabase";

export const AUTH_MODE: AuthMode =
  process.env.NEXT_PUBLIC_AUTH_MODE === "supabase" ? "supabase" : "demo";

let client: SupabaseClient | null = null;

export function supabase(): SupabaseClient | null {
  if (AUTH_MODE !== "supabase") return null;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return null;
  client ??= createClient(url, key, { auth: { persistSession: true, autoRefreshToken: true } });
  return client;
}

export async function accessToken(): Promise<string | null> {
  const sb = supabase();
  if (!sb) return null;
  const { data } = await sb.auth.getSession();
  return data.session?.access_token ?? null;
}

export async function authHeaders(tenantId?: string): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  const token = await accessToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (tenantId) headers["X-Tenant-ID"] = tenantId;
  return headers;
}
