import { AUTHORITY, type Authority } from "./content";

function Node({
  title,
  sub,
  authority,
  className = "",
}: {
  title: string;
  sub?: string | string[]; // an array renders one line per entry
  authority?: Authority;
  className?: string;
}) {
  const a = authority ? AUTHORITY[authority] : null;
  return (
    <div className={`rounded-xl bg-white px-3 py-2.5 text-center shadow-sm ring-1 ring-slate-200 ${className}`}>
      <p className="text-sm font-semibold text-slate-900">
        {a && (
          <span aria-hidden className="mr-1 text-slate-500">
            {a.mark}
          </span>
        )}
        {title}
      </p>
      {(Array.isArray(sub) ? sub : sub ? [sub] : []).map((line) => (
        <p key={line} className="mt-0.5 text-xs text-slate-500">
          {line}
        </p>
      ))}
    </div>
  );
}

function Down({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-1" aria-hidden>
      <span className="h-4 w-px bg-slate-300" />
      {label && <span className="px-2 text-[11px] text-slate-500">{label}</span>}
      <span className="text-xs leading-none text-slate-400">▼</span>
    </div>
  );
}

function Group({ label, children, tone = "border-slate-200 bg-slate-50/70" }: { label: string; children: React.ReactNode; tone?: string }) {
  return (
    <div className={`rounded-2xl border border-dashed p-3 ${tone}`}>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
      {children}
    </div>
  );
}

/** Real deployed topology (docs/DEPLOYMENT.md, backend/app/agent/graph). Pure HTML/CSS. */
export default function ArchitectureDiagram() {
  return (
    <figure aria-labelledby="architecture-caption" className="mx-auto w-full max-w-4xl" data-testid="architecture-diagram">
      <div className="grid gap-2 sm:grid-cols-[1fr_auto_14rem] sm:items-center">
        <Node title="Browser · Next.js" sub="Vercel · chat, citations, approval cards, trace" />
        <span aria-hidden className="hidden text-center text-slate-400 sm:block">⇄</span>
        <Node title="Supabase Auth" sub="JWT (reviewer or anonymous demo)" />
      </div>
      <Down label="HTTPS · Bearer JWT · tenant selector" />
      <Group label="Railway · FastAPI (modular monolith)">
        <Node
          title="Verified principal → trusted tenant"
          sub="JWKS verification · memberships or read-only demo tenant · rate limits"
          authority="deterministic"
        />
        <Down />
        <div className="grid gap-2 sm:grid-cols-[1fr_12rem] sm:items-center">
          <Node title="LangGraph orchestration" sub="MODEL · TOOLS · RETRIEVE · PROPOSE · APPROVAL · EXECUTE" authority="deterministic" />
          <Node
            title="LLM provider layer"
            sub={["Hosted: Cloudflare Workers AI → Gemini fallback per call", "Local dev: Ollama"]}
            authority="model"
          />
        </div>
        <Down />
        <div className="grid gap-2 md:grid-cols-3">
          <Group label="Business facts" tone="border-slate-300 bg-white">
            <Node title="12 commerce tools" sub="typed LangChain tools, tenant injected" authority="deterministic" />
          </Group>
          <Group label="Policy knowledge" tone="border-sky-200 bg-sky-50/50">
            <Node title="Policy retrieval" sub="tenant + effective date · citations validated" authority="knowledge" />
          </Group>
          <Group label="Write boundary" tone="border-amber-400 bg-amber-50/60">
            <Node title="Approval-gated actions" sub="propose → human → deterministic execute" authority="human" />
          </Group>
        </div>
      </Group>
      <Down />
      <Group label="Supabase · PostgreSQL + pgvector">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Node title="Commerce tables" sub="composite tenant FKs" />
          <Node
            title="Policy chunks + vectors"
            sub={["pgvector · one profile per embedding model", "Hosted: Gemini · Local dev: Ollama"]}
          />
          <Node title="Actions + audit events" sub="idempotency keys, same transaction" />
          <Node title="LangGraph checkpoints" sub="durable pause / resume" />
        </div>
      </Group>
      <Down />
      <Node title="Grounded answer + citations + execution trace" className="mx-auto max-w-md" />
      <figcaption id="architecture-caption" className="mt-4 flex flex-wrap justify-center gap-3 text-xs text-slate-600">
        {(Object.keys(AUTHORITY) as Authority[]).map((k) => (
          <span key={k}>
            <span aria-hidden className="mr-1 font-semibold">
              {AUTHORITY[k].mark}
            </span>
            {AUTHORITY[k].label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
