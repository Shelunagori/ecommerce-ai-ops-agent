import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CommerceOps AI",
  description: "Ecommerce AI Operations Agent",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-slate-50">{children}</body>
    </html>
  );
}
