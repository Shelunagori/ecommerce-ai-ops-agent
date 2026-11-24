import type { Metadata } from "next";
import "./globals.css";

import BackendWakeNotice from "@/components/BackendWakeNotice";

export const metadata: Metadata = {
  title: "CommerceOps AI",
  description: "Ecommerce AI operations assistant with approval-gated actions (synthetic data)",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-slate-50">
        {children}
        <BackendWakeNotice />
      </body>
    </html>
  );
}
