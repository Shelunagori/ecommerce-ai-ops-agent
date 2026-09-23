import StatusPanel from "@/components/StatusPanel";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-6 py-16">
      <header>
        <p className="text-sm font-medium uppercase tracking-wide text-indigo-600">Foundation</p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight text-slate-900">CommerceOps AI</h1>
        <p className="mt-2 text-lg text-slate-600">Ecommerce AI Operations Agent</p>
      </header>

      <StatusPanel />

      <section className="rounded-xl border border-dashed border-slate-300 p-6 text-sm text-slate-600">
        <h2 className="font-semibold text-slate-800">What&apos;s next</h2>
        <p className="mt-2">
          This is the project foundation. Agent functionality — working with customers, orders,
          invoices, shipments and company policies — will be added in later steps. Nothing
          here uses real customer or company data.
        </p>
      </section>
    </main>
  );
}
