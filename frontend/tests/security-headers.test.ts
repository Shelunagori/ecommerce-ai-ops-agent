import { describe, expect, it } from "vitest";

import { contentSecurityPolicy, securityHeaders } from "../security-headers";

describe("security headers", () => {
  it("limits network access to self, the API and Supabase", () => {
    const csp = contentSecurityPolicy({
      apiUrl: "https://api.example.app/",
      supabaseUrl: "https://demo.supabase.co",
      isDev: false,
    });
    expect(csp).toContain("connect-src 'self' https://api.example.app https://demo.supabase.co;");
    expect(csp).toContain("frame-ancestors 'none'");
    expect(csp).toContain("object-src 'none'");
    expect(csp).not.toContain("unsafe-eval");
    expect(csp).toContain("upgrade-insecure-requests");
  });

  it("keeps a local http backend working and allows dev tooling only in dev", () => {
    const csp = contentSecurityPolicy({ apiUrl: "http://localhost:8000", isDev: true });
    expect(csp).toContain("connect-src 'self' http://localhost:8000 ws: wss:");
    expect(csp).not.toContain("upgrade-insecure-requests");
    expect(csp).toContain("'unsafe-eval'");
  });

  it("ignores malformed URLs instead of widening the policy", () => {
    const csp = contentSecurityPolicy({ apiUrl: "not a url", isDev: false });
    expect(csp).toContain("connect-src 'self';");
  });

  it("sends the standard hardening headers", () => {
    const keys = securityHeaders({ isDev: false }).map((h) => h.key);
    expect(keys).toEqual([
      "Content-Security-Policy",
      "X-Content-Type-Options",
      "X-Frame-Options",
      "Referrer-Policy",
      "Permissions-Policy",
    ]);
  });
});
