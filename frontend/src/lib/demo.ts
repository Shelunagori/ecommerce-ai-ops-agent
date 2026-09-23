/**
 * Demo-only configuration: the synthetic tenants created by backend/scripts/seed_demo.py.
 *
 * The ids are deterministic UUID5 values produced by the seed script, so they are the
 * same on every machine. They are not secrets. Sending them as X-Tenant-ID is a
 * temporary demo mechanism for tenant context — it is NOT authentication.
 */
export type DemoTenant = { id: string; name: string };

export const DEMO_TENANTS: readonly DemoTenant[] = [
  { id: "17243d88-ed66-5445-955b-7d7572094122", name: "Northstar Commerce" },
  { id: "11a6d918-f89f-59b5-ab1e-45a109978c8a", name: "BluePeak Retail" },
];

export const DEFAULT_DEMO_TENANT = DEMO_TENANTS[0];
