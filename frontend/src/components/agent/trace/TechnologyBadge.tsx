import type { ExecutionTraceEvent } from "@/lib/types";

const PROVIDERS: Record<string, string> = { gemini: "Gemini", ollama: "Ollama" };

/**
 * Technology badges derived ONLY from what the event says actually happened (its kind and
 * safe metadata such as the retriever identity or model provider) — never decoration.
 */
export function badgesFor(event: ExecutionTraceEvent): string[] {
  const m = event.metadata;
  switch (event.kind) {
    case "request":
      return ["FastAPI", "Verified tenant scope"];
    case "model": {
      const provider = PROVIDERS[String(m.provider ?? "")];
      return provider ? ["LangGraph", provider] : ["LangGraph"];
    }
    case "commerce_tool":
      return ["LangChain Tool", "PostgreSQL"];
    case "retrieval": {
      const retriever = String(m.retriever ?? "");
      if (retriever.startsWith("semantic-pgvector")) return ["RAG", "pgvector", "PostgreSQL"];
      if (retriever.startsWith("lexical")) return ["RAG", "PostgreSQL full-text"];
      return ["RAG"];
    }
    case "grounding":
      return ["Citation validation"];
    case "action_proposal":
    case "approval":
      return ["HITL"];
    case "checkpoint":
      return m.durable ? ["PostgreSQL Checkpoint"] : ["Checkpoint (in-memory)"];
    case "action_execution":
      return ["PostgreSQL", "Audit event"];
    default:
      return [];
  }
}

export default function TechnologyBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] font-medium text-slate-600">
      {label}
    </span>
  );
}
