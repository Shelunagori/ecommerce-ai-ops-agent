import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { resetBackendWake } from "@/lib/backendWake";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  resetBackendWake();
});
