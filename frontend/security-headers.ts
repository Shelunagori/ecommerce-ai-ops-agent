/**
 * Response security headers for the frontend (Phase 12), applied by next.config.ts.
 *
 * CSP without nonces (Next.js guide "Without Nonces"): scripts/styles from 'self' (+ the
 * inline bootstrap Next.js needs), network access only to this origin, the configured API
 * and the configured Supabase project; no framing, no plugins, no foreign form targets.
 */
function origin(url: string | undefined): string | null {
  if (!url) return null;
  try {
    return new URL(url).origin;
  } catch {
    return null;
  }
}

export function contentSecurityPolicy(opts: { apiUrl?: string; supabaseUrl?: string; isDev: boolean }): string {
  const api = origin(opts.apiUrl);
  const supabase = origin(opts.supabaseUrl);
  const connect = ["'self'", api, supabase].filter(Boolean).join(" ");
  const directives = [
    "default-src 'self'",
    `script-src 'self' 'unsafe-inline'${opts.isDev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' blob: data:",
    "font-src 'self'",
    `connect-src ${connect}${opts.isDev ? " ws: wss:" : ""}`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ];
  // Only when the API itself is HTTPS: never break a local http://localhost backend.
  if (api?.startsWith("https://")) directives.push("upgrade-insecure-requests");
  return directives.join("; ");
}

export function securityHeaders(opts: { apiUrl?: string; supabaseUrl?: string; isDev: boolean }) {
  return [
    { key: "Content-Security-Policy", value: contentSecurityPolicy(opts) },
    { key: "X-Content-Type-Options", value: "nosniff" },
    { key: "X-Frame-Options", value: "DENY" },
    { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  ];
}
