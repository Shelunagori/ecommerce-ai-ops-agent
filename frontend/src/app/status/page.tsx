import DemoDataPanel from "@/components/DemoDataPanel";
import StatusPanel from "@/components/StatusPanel";

export default function StatusPage() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-6 py-16">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">System status</h1>
        <p className="mt-1 text-sm text-slate-600">API, database and synthetic demo data.</p>
      </header>
      <StatusPanel />
      <DemoDataPanel />
    </main>
  );
}
