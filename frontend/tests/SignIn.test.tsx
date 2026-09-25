import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Session = { access_token: string } | null;
const auth = {
  session: null as Session,
  getSession: vi.fn(async () => ({ data: { session: auth.session } })),
  signInAnonymously: vi.fn(async () => ({ data: {}, error: null as null | { message: string } })),
  signInWithPassword: vi.fn(async () => ({ data: {}, error: null })),
};

vi.mock("@supabase/supabase-js", () => ({
  createClient: () => ({ auth }),
}));

async function renderLanding() {
  vi.resetModules();
  const { default: SignIn } = await import("@/components/agent/SignIn");
  render(<SignIn />);
}

describe("Auth landing", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_AUTH_MODE", "supabase");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_URL", "https://demo-project.supabase.co");
    vi.stubEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "public-anon-key-for-tests");
    auth.session = null;
    auth.getSession.mockClear();
    auth.signInAnonymously.mockReset().mockResolvedValue({ data: {}, error: null });
    auth.signInWithPassword.mockClear();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    window.history.replaceState(null, "", "/");
  });

  it("shows both entry points, the review link and no credentials", async () => {
    await renderLanding();
    expect(screen.getByRole("heading", { name: "CommerceOps AI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try Live Demo" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reviewer Sign In" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("link", { name: "Review the Engineering →" })).toHaveAttribute("href", "/review");
    expect(screen.getByText("Synthetic data only.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/public-anon-key|password:|service_role/i);
  });

  it("Try Live Demo creates an anonymous session", async () => {
    await renderLanding();
    await userEvent.click(screen.getByRole("button", { name: "Try Live Demo" }));
    await waitFor(() => expect(auth.signInAnonymously).toHaveBeenCalledTimes(1));
  });

  it("reuses an existing session instead of creating another anonymous user", async () => {
    auth.session = { access_token: "existing" };
    await renderLanding();
    await userEvent.click(screen.getByRole("button", { name: "Try Live Demo" }));
    await waitFor(() => expect(auth.getSession).toHaveBeenCalled());
    expect(auth.signInAnonymously).not.toHaveBeenCalled();
  });

  it("explains a disabled anonymous sign-in and can retry", async () => {
    auth.signInAnonymously.mockResolvedValueOnce({ data: {}, error: { message: "Anonymous sign-ins are disabled" } });
    await renderLanding();
    await userEvent.click(screen.getByRole("button", { name: "Try Live Demo" }));
    expect(await screen.findByText(/public demo is not available right now/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(auth.signInAnonymously).toHaveBeenCalledTimes(2));
  });

  it("Reviewer Sign In reveals the email/password form and signs in", async () => {
    await renderLanding();
    await userEvent.click(screen.getByRole("button", { name: "Reviewer Sign In" }));
    await userEvent.type(screen.getByLabelText("Email"), "reviewer@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "not-a-real-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() =>
      expect(auth.signInWithPassword).toHaveBeenCalledWith({ email: "reviewer@example.com", password: "not-a-real-password" }),
    );
    expect(auth.signInAnonymously).not.toHaveBeenCalled();
  });

  it("?demo=1 (from /review) starts the demo directly", async () => {
    window.history.replaceState(null, "", "/?demo=1");
    await renderLanding();
    await waitFor(() => expect(auth.signInAnonymously).toHaveBeenCalledTimes(1));
    expect(window.location.search).toBe("");
  });
});
