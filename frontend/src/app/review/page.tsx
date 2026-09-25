import type { Metadata } from "next";
import Link from "next/link";

import ArchitectureDiagram from "@/components/review/ArchitectureDiagram";
import {
  AUTHORITY,
  DECISIONS,
  DEPLOYMENT,
  HITL_STEPS,
  RAG_STEPS,
  REPO_URL,
  SAFETY,
  SCENARIOS,
} from "@/components/review/content";
import { EVIDENCE, EVIDENCE_AS_OF } from "@/components/review/evidence";
import FlowWalkthrough from "@/components/review/FlowWalkthrough";

export const metadata: Metadata = {
  title: "Engineering review · CommerceOps AI",
  description:
    "How CommerceOps AI works: LangGraph orchestration, deterministic tools, pgvector RAG with validated citations, human-approved writes and durable state.",
};

const NAV = [
  ["overview", "Overview"],
  ["architecture", "Architecture"],
  ["agent-flow", "Agent Flow"],
  ["rag", "RAG"],
  ["hitl", "HITL"],
  ["security", "Security"],
  ["evaluation", "Evaluation"],
  ["deployment", "Deployment"],
] as const;

function Section({ id, title, lead, children }: { id: string; title: string; lead?: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-20 border-t border-slate-200 py-12 sm:py-16">
      <h2 id={`${id}-title`} className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
        {title}
      </h2>
      {lead && <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-600">{lead}</p>}
      <div className="mt-6">{children}</div>
    </section>
  );
}

function Card({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      <p className="mt-1 text-sm leading-relaxed text-slate-600">{body}</p>
    </div>
  );
}

function Cta({ href, children, primary = false, external = false }: { href: string; children: React.ReactNode; primary?: boolean; external?: boolean }) {
  const cls = `inline-flex items-center rounded-lg px-4 py-2 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 ${
    primary ? "bg-indigo-600 text-white shadow-sm hover:bg-indigo-700" : "bg-white text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50"
  }`;
  return external ? (
    <a href={href} className={cls} target="_blank" rel="noreferrer">
      {children}
    </a>
  ) : (
    <Link href={href} className={cls}>
      {children}
    </Link>
  );
}

function Chain({ steps }: { steps: { text: string; tone?: string; mark?: string }[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-1 gap-y-2 text-sm">
      {steps.map((s, i) => (
        <li key={s.text} className="flex items-center gap-1">
          <span className={`rounded-lg px-2.5 py-1.5 ring-1 ring-inset ${s.tone ?? "bg-white text-slate-800 ring-slate-200"}`}>
            {s.mark && (
              <span aria-hidden className="mr-1 font-semibold">
                {s.mark}
              </span>
            )}
            {s.text}
          </span>
          {i < steps.length - 1 && (
            <span aria-hidden className="text-slate-400">
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

export default function ReviewPage() {
  return (
    <div className="flex-1 bg-slate-50">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
          <Link href="/" className="mr-auto text-sm font-bold tracking-tight text-slate-900">
            CommerceOps AI <span className="font-normal text-slate-500">· Engineering review</span>
          </Link>
          <nav aria-label="Review sections" className="order-last w-full overflow-x-auto sm:order-none sm:w-auto">
            <ul className="flex gap-3 whitespace-nowrap text-xs font-medium text-slate-600">
              {NAV.map(([id, label]) => (
                <li key={id}>
                  <a href={`#${id}`} className="hover:text-slate-900 focus:outline-none focus-visible:underline">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
          <Link href="/?demo=1" className="rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-700">
            Try Live Demo
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 sm:px-6">
        <section id="overview" aria-labelledby="overview-title" className="scroll-mt-20 py-12 sm:py-20">
          <p className="text-xs font-semibold uppercase tracking-widest text-indigo-600">Engineering case study</p>
          <h1 id="overview-title" className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            CommerceOps AI
          </h1>
          <p className="mt-1 text-lg text-slate-700">Production-grade Ecommerce AI Operations Agent</p>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-600 sm:text-base">
            A multi-tenant operations agent: a LangGraph orchestrator calls deterministic, tenant-scoped commerce tools for exact
            facts, retrieves versioned policies with pgvector and must cite them, and can only <em>propose</em> writes — a human
            approves before deterministic code executes them once. Conversations and pending approvals live in durable PostgreSQL
            checkpoints. All data is synthetic.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Cta href="/?demo=1" primary>
              Try Live Demo
            </Cta>
            <Cta href="/">Open Application</Cta>
            <Cta href={REPO_URL} external>
              View GitHub
            </Cta>
          </div>
          <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {SCENARIOS.map((s) => (
              <Card key={s.title} {...s} />
            ))}
          </div>
        </section>

        <Section
          id="architecture"
          title="Architecture"
          lead="One FastAPI service, one PostgreSQL. The trusted principal — not the model — decides the tenant; business facts, policy knowledge and writes are separate lanes."
        >
          <ArchitectureDiagram />
        </Section>

        <Section
          id="agent-flow"
          title="Agent flow"
          lead="The graph paths behind four request types. In the app, every answer carries the backend-recorded execution trace of the run that produced it."
        >
          <FlowWalkthrough />
        </Section>

        <Section
          id="rag"
          title="RAG design"
          lead="Policies are versioned with effective dates. Retrieval is filtered by the trusted tenant and the requested date, and an answer may only cite chunks retrieved in the current turn."
        >
          <Chain steps={RAG_STEPS.map((text) => ({ text, tone: AUTHORITY.knowledge.tone }))} />
          <p className="mt-4 text-sm text-slate-600">
            Each embedding provider, model revision and dimension is its own profile: vectors from different models are never
            compared, and a missing profile fails closed instead of falling back.
          </p>
        </Section>

        <Section
          id="hitl"
          title="Human-in-the-loop"
          lead="The LLM never performs the mutation. It can only create a pending request; the graph pauses; an approver decides; application code executes it exactly once."
        >
          <Chain steps={HITL_STEPS.map((s) => ({ text: s.text, tone: AUTHORITY[s.authority].tone, mark: AUTHORITY[s.authority].mark }))} />
          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            <Card
              title="Public demo is read-only by construction"
              body="Anonymous visitors get a graph profile with no action tools and no access to action resources — not just hidden buttons."
            />
            <Card title="Bound to what the approver saw" body="Approval carries the arguments hash shown on the card; a changed action needs a new approval." />
          </div>
        </Section>

        <Section
          id="security"
          title="Security & safety boundaries"
          lead="Enforced in code and covered by tests (including mutation tests that must fail when a guard is removed)."
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {SAFETY.map((s) => (
              <Card key={s.title} {...s} />
            ))}
          </div>
          <div className="mt-8" data-testid="tenant-isolation">
            <h3 className="text-sm font-semibold text-slate-900">Tenant isolation</h3>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {[
                ["Northstar Commerce", "Own customers, orders ORD-1001…, policies"],
                ["BluePeak Retail", "Same order numbers, different rows — and the public demo tenant"],
              ].map(([name, body]) => (
                <div key={name} className="rounded-xl border-2 border-slate-300 bg-white p-4">
                  <p className="text-sm font-semibold text-slate-900">{name}</p>
                  <p className="mt-1 text-xs text-slate-600">{body}</p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-sm text-slate-600">
              The verified principal selects the tenant; the database rejects cross-tenant references. Demo visitors share
              BluePeak&apos;s synthetic data but never a conversation: threads are keyed by the visitor&apos;s own identity.
            </p>
          </div>
        </Section>

        <Section id="evaluation" title="Evaluation & testing" lead={`Measured on this codebase (${EVIDENCE_AS_OF}).`}>
          <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {EVIDENCE.map((e) => (
              <div key={e.label} className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
                <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{e.label}</dt>
                <dd className="mt-1 text-lg font-bold text-slate-900">{e.value}</dd>
                <dd className="mt-1 text-xs text-slate-600">{e.note}</dd>
              </div>
            ))}
          </dl>
        </Section>

        <Section
          id="deployment"
          title="Deployment topology"
          lead="A push to main runs GitHub Actions CI while Railway and Vercel deploy through their native Git integrations."
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {DEPLOYMENT.map((d) => (
              <Card key={d.name} title={d.name} body={d.role} />
            ))}
          </div>
        </Section>

        <Section id="decisions" title="Design decisions">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {DECISIONS.map((d) => (
              <Card key={d.title} {...d} />
            ))}
          </div>
        </Section>

        <section className="border-t border-slate-200 py-16 text-center">
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">Try the live system</h2>
          <p className="mt-2 text-sm text-slate-600">No sign-up: a read-only demo session on synthetic data.</p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <Cta href="/?demo=1" primary>
              Launch Live Demo
            </Cta>
            <Cta href={REPO_URL} external>
              View GitHub
            </Cta>
          </div>
        </section>
      </main>
    </div>
  );
}
