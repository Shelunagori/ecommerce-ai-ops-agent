"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { ARCH_TOURS, type ArchNode, AUTHORITY, type Authority, FLOWS } from "./content";

/*
 * Real deployed topology (docs/DEPLOYMENT.md, backend/app/agent/graph) drawn as one SVG graph
 * on a fixed grid: right-angle connectors with arrowheads, labels on the edges and dots that
 * flow along every connector. A request "packet" travels ONLY along the connectors, hop by
 * hop, on the real graph path of the selected flow (ARCH_TOURS). Plays when scrolled into
 * view; prefers-reduced-motion gets the same steps without motion and without autoplay.
 */

// ---------------------------------------------------------------------------------------------
// Layout (viewBox units)

const VIEW_W = 1160;
const VIEW_H = 462;
const NODE_H = 64;

/** What a box is, for its color and the filter chips. */
type Kind = "plain" | "model" | "authority" | "knowledge" | "human";

type NodeDef = { id: ArchNode; x: number; y: number; w: number; title: string; sub: string; kind: Kind };

const NODES: NodeDef[] = [
  { id: "browser", x: 20, y: 50, w: 190, title: "Browser", sub: "Next.js · Vercel", kind: "plain" },
  { id: "principal", x: 270, y: 50, w: 200, title: "Verified principal", sub: "JWKS → trusted tenant", kind: "authority" },
  { id: "graph", x: 530, y: 50, w: 200, title: "LangGraph orchestration", sub: "graph nodes · routing · checkpoints", kind: "authority" },
  { id: "llm", x: 800, y: 50, w: 200, title: "LLM provider layer", sub: "Cloudflare AI → Gemini fallback", kind: "model" },
  { id: "human", x: 1030, y: 50, w: 120, title: "Human approver", sub: "approve / reject", kind: "human" },
  { id: "auth", x: 20, y: 210, w: 190, title: "Supabase Auth", sub: "JWT · reviewer or anon demo", kind: "plain" },
  { id: "tools", x: 340, y: 210, w: 190, title: "12 commerce tools", sub: "typed · tenant injected", kind: "plain" },
  { id: "retrieval", x: 560, y: 210, w: 190, title: "Policy retrieval", sub: "tenant + effective date", kind: "knowledge" },
  { id: "approval", x: 780, y: 210, w: 190, title: "Approval-gated actions", sub: "propose → human → execute", kind: "authority" },
  { id: "dbCommerce", x: 340, y: 370, w: 190, title: "Commerce tables", sub: "PostgreSQL · tenant FKs", kind: "plain" },
  { id: "dbVectors", x: 560, y: 370, w: 190, title: "Policy chunks + vectors", sub: "pgvector · Gemini embeddings", kind: "plain" },
  { id: "dbAudit", x: 780, y: 370, w: 190, title: "Actions + audit events", sub: "idempotency keys", kind: "plain" },
  { id: "dbCheckpoints", x: 990, y: 370, w: 160, title: "LangGraph checkpoints", sub: "durable pause / resume", kind: "plain" },
];
const NODE_BY_ID = Object.fromEntries(NODES.map((n) => [n.id, n])) as Record<ArchNode, NodeDef>;

type P = [number, number];
type EdgeDef = {
  id: string;
  from: ArchNode;
  to: ArchNode;
  pts: P[];
  label: string;
  lx: number;
  ly: number;
  both?: boolean; // arrowheads at both ends
  ret?: boolean; // a return path (dashed)
  mid?: boolean; // label centered on (lx, ly)
};

const EDGES: EdgeDef[] = [
  { id: "signin", from: "browser", to: "auth", pts: [[115, 114], [115, 210]], label: "sign-in", lx: 123, ly: 166, both: true },
  { id: "jwks", from: "auth", to: "principal", pts: [[210, 242], [300, 242], [300, 114]], label: "JWKS", lx: 222, ly: 235 },
  { id: "request", from: "browser", to: "principal", pts: [[210, 82], [270, 82]], label: "JWT", lx: 240, ly: 75, mid: true },
  { id: "tenant", from: "principal", to: "graph", pts: [[470, 82], [530, 82]], label: "tenant", lx: 500, ly: 75, mid: true },
  { id: "model", from: "graph", to: "llm", pts: [[730, 82], [800, 82]], label: "propose", lx: 765, ly: 75, both: true, mid: true },
  { id: "toolcall", from: "graph", to: "tools", pts: [[630, 114], [630, 160], [435, 160], [435, 210]], label: "tool call", lx: 443, ly: 192 },
  { id: "retrieve", from: "graph", to: "retrieval", pts: [[630, 114], [630, 160], [655, 160], [655, 210]], label: "retrieve", lx: 663, ly: 192 },
  { id: "action", from: "graph", to: "approval", pts: [[630, 114], [630, 160], [875, 160], [875, 210]], label: "action", lx: 883, ly: 192 },
  { id: "sql", from: "tools", to: "dbCommerce", pts: [[435, 274], [435, 370]], label: "SQL", lx: 443, ly: 322 },
  { id: "cosine", from: "retrieval", to: "dbVectors", pts: [[655, 274], [655, 370]], label: "cosine", lx: 663, ly: 322 },
  { id: "audit", from: "approval", to: "dbAudit", pts: [[875, 274], [875, 370]], label: "audit", lx: 883, ly: 322 },
  { id: "checkpoint", from: "approval", to: "dbCheckpoints", pts: [[935, 274], [935, 322], [1070, 322], [1070, 370]], label: "checkpoint", lx: 985, ly: 315 },
  { id: "execute", from: "approval", to: "dbCommerce", pts: [[815, 274], [815, 338], [480, 338], [480, 370]], label: "execute · 1 write", lx: 690, ly: 356 },
  { id: "decision", from: "approval", to: "human", pts: [[970, 242], [1090, 242], [1090, 114]], label: "decision", lx: 1000, ly: 235, both: true },
  { id: "answer", from: "graph", to: "browser", pts: [[630, 50], [630, 22], [115, 22], [115, 50]], label: "answer + execution trace (SSE)", lx: 372, ly: 16, ret: true, mid: true },
];

const KIND_STROKE: Record<Kind, string> = {
  plain: "var(--arch-plain)",
  model: "var(--arch-model)",
  authority: "var(--arch-authority)",
  knowledge: "var(--arch-knowledge)",
  human: "var(--arch-human)",
};
const AUTHORITY_COLOR: Record<Authority, string> = {
  deterministic: "var(--arch-authority)",
  model: "var(--arch-model)",
  knowledge: "var(--arch-knowledge)",
  human: "var(--arch-human)",
};

const FILTERS: Array<{ id: "all" | Kind; label: string }> = [
  { id: "all", label: "All" },
  { id: "model", label: "Language model" },
  { id: "authority", label: "Authority" },
  { id: "knowledge", label: "Knowledge" },
  { id: "human", label: "Human" },
];

// ---------------------------------------------------------------------------------------------
// Routing: the packet moves along connectors only (shortest route between two boxes).

type Step = { edge: EdgeDef; forward: boolean };

function route(from: ArchNode, to: ArchNode): Step[] {
  if (from === to) return [];
  const prev = new Map<ArchNode, Step & { node: ArchNode }>();
  const seen = new Set<ArchNode>([from]);
  const queue: ArchNode[] = [from];
  while (queue.length) {
    const at = queue.shift()!;
    for (const edge of EDGES) {
      const forward = edge.from === at;
      if (!forward && edge.to !== at) continue;
      const next = forward ? edge.to : edge.from;
      if (seen.has(next)) continue;
      seen.add(next);
      prev.set(next, { edge, forward, node: at });
      if (next === to) {
        const steps: Step[] = [];
        for (let n: ArchNode = to; n !== from; ) {
          const p = prev.get(n)!;
          steps.unshift({ edge: p.edge, forward: p.forward });
          n = p.node;
        }
        return steps;
      }
      queue.push(next);
    }
  }
  return [];
}

function routePoints(steps: Step[]): P[] {
  return steps.flatMap(({ edge, forward }) => (forward ? edge.pts : [...edge.pts].reverse()));
}

function polylineLength(pts: P[]): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return len;
}

function pointAt(pts: P[], t: number): P {
  const total = polylineLength(pts);
  let left = total * t;
  for (let i = 1; i < pts.length; i++) {
    const seg = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    if (left <= seg && seg > 0) {
      const k = left / seg;
      return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * k, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * k];
    }
    left -= seg;
  }
  return pts[pts.length - 1];
}

const toPath = (pts: P[]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
/** Resting point before any travel: the box's top edge, so the packet never covers its text. */
const topEdge = (id: ArchNode): P => {
  const n = NODE_BY_ID[id];
  return [n.x + n.w / 2, n.y];
};

// ---------------------------------------------------------------------------------------------

const DWELL_MS = 1_300;
const PAUSE_DWELL_MS = 2_800; // the human approval wait lingers
const END_HOLD_MS = 2_400;
const TRAVEL_PX_PER_MS = 0.55;
const travelMs = (len: number) => Math.min(1_500, Math.max(450, len / TRAVEL_PX_PER_MS));
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function subscribeReducedMotion(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}
const reducedMotionNow = () =>
  typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const useReducedMotion = () => useSyncExternalStore(subscribeReducedMotion, reducedMotionNow, () => false);

type NodeState = "active" | "visited" | "idle" | "dim";

export default function ArchitectureDiagram() {
  const reducedMotion = useReducedMotion();
  const [flowIdx, setFlowIdx] = useState(0);
  const [hop, setHop] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [userPressedPlay, setUserPressedPlay] = useState(false); // reduced motion: only on request
  const [inView, setInView] = useState(false);
  const [filter, setFilter] = useState<"all" | Kind>("all");

  const cardRef = useRef<HTMLDivElement | null>(null);
  const packetRef = useRef<SVGGElement | null>(null);

  const flow = FLOWS[flowIdx];
  const tour = ARCH_TOURS[flow.id];
  const current = tour[hop];
  const atEnd = hop === tour.length - 1;
  const isPlaying = playing && (!reducedMotion || userPressedPlay) && filter === "all";

  // The connectors this hop travels, and every connector the flow has used so far.
  const hopSteps = useMemo(() => (hop === 0 ? [] : route(tour[hop - 1].node, current.node)), [tour, hop, current.node]);
  const usedEdges = useMemo(() => {
    const used = new Set<string>();
    for (let i = 1; i < hop; i++) route(tour[i - 1].node, tour[i].node).forEach((s) => used.add(s.edge.id));
    return used;
  }, [tour, hop]);
  const activeEdges = useMemo(() => new Map(hopSteps.map((s) => [s.edge.id, s.forward])), [hopSteps]);
  const travel = useMemo(() => {
    const pts = routePoints(hopSteps);
    return { pts, ms: pts.length > 1 ? travelMs(polylineLength(pts)) : 0 };
  }, [hopSteps]);

  // Packet: glides along the connectors (one rAF loop per hop; no React re-render per frame).
  useEffect(() => {
    const el = packetRef.current;
    if (!el) return;
    const place = ([x, y]: P) => el.setAttribute("transform", `translate(${x} ${y})`);
    // The route ends on the connector's anchor at the box's border: the packet rests there.
    if (travel.pts.length < 2 || reducedMotion || typeof requestAnimationFrame === "undefined") {
      place(travel.pts.length ? travel.pts[travel.pts.length - 1] : topEdge(current.node));
      return;
    }
    const pts = travel.pts;
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / travel.ms);
      place(pointAt(pts, ease(t)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [travel, current.node, reducedMotion]);

  // Play only while the diagram is on screen.
  useEffect(() => {
    const el = cardRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Autoplay: travel + dwell, then the next hop; after the last hop, the next flow.
  useEffect(() => {
    if (!isPlaying || !inView) return;
    const wait = atEnd ? END_HOLD_MS : travel.ms + (current.pause ? PAUSE_DWELL_MS : DWELL_MS);
    const id = setTimeout(() => {
      if (atEnd) {
        setFlowIdx((i) => (i + 1) % FLOWS.length);
        setHop(0);
      } else {
        setHop((h) => h + 1);
      }
    }, wait);
    return () => clearTimeout(id);
  }, [isPlaying, inView, atEnd, current.pause, travel.ms, hop, flowIdx]);

  const selectFlow = useCallback((i: number) => {
    setFlowIdx(i);
    setHop(0);
    setFilter("all");
  }, []);
  const togglePlay = () => {
    if (isPlaying) return setPlaying(false);
    setUserPressedPlay(true);
    setPlaying(true);
    setFilter("all");
    if (atEnd) setHop(0);
  };
  const step = (delta: number) => {
    setPlaying(false);
    setFilter("all");
    setHop((h) => Math.min(tour.length - 1, Math.max(0, h + delta)));
  };

  const onPath = new Set(tour.map((t) => t.node));
  const visited = new Set(tour.slice(0, hop).map((t) => t.node));
  const stateOf = (n: NodeDef): NodeState => {
    if (filter !== "all") return n.kind === filter ? "active" : "dim";
    if (n.id === current.node) return "active";
    if (visited.has(n.id)) return "visited";
    return onPath.has(n.id) ? "idle" : "dim";
  };
  const hopColor = AUTHORITY_COLOR[current.authority];

  return (
    <figure aria-labelledby="architecture-caption" className="mx-auto w-full max-w-6xl" data-testid="architecture-diagram">
      <div ref={cardRef} className="arch-card rounded-3xl p-4 sm:p-6">
        {/* Header: title + filter chips */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-lg font-semibold tracking-tight text-slate-900 sm:text-xl">Request path</h3>
          <div role="group" aria-label="Highlight components" className="arch-chips flex flex-wrap gap-1 rounded-full p-1">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                aria-pressed={filter === f.id}
                onClick={() => setFilter(f.id)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 sm:text-sm ${
                  filter === f.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Flow tabs + playback */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <div role="tablist" aria-label="Animated request flows" className="flex flex-wrap gap-2">
            {FLOWS.map((f, i) => (
              <button
                key={f.id}
                type="button"
                role="tab"
                aria-selected={i === flowIdx}
                onClick={() => selectFlow(i)}
                className={`rounded-full px-3 py-1.5 text-sm font-medium ring-1 ring-inset transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  i === flowIdx ? "bg-slate-900 text-white ring-slate-900" : "bg-white/70 text-slate-700 ring-slate-300 hover:bg-white"
                }`}
              >
                {f.tab}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-1">
            <button type="button" onClick={() => step(-1)} disabled={hop === 0} aria-label="Previous step" className="arch-ctl">
              ◀
            </button>
            <button type="button" onClick={togglePlay} aria-label={isPlaying ? "Pause animation" : "Play animation"} className="arch-ctl min-w-20">
              {isPlaying ? "❚❚ Pause" : "▶ Play"}
            </button>
            <button type="button" onClick={() => step(1)} disabled={atEnd} aria-label="Next step" className="arch-ctl">
              ▶
            </button>
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-600">
          <span className="font-medium text-slate-900">Request:</span> “{flow.prompt}”
        </p>

        {/* The graph */}
        <div className="-mx-4 mt-3 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <svg
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            className="arch-svg block h-auto w-full min-w-[760px]"
            role="img"
            aria-label="Architecture: browser, Supabase Auth, verified principal, LangGraph, LLM provider, commerce tools, policy retrieval, approval-gated actions, human approver and the PostgreSQL stores"
          >
            <defs>
              <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0 0 L10 5 L0 10 z" fill="context-stroke" />
              </marker>
            </defs>

            {/* Hosting frames */}
            <rect x={255} y={36} width={760} height={258} rx={18} className="arch-frame" />
            <text x={267} y={286} className="arch-frame-label">
              Render · FastAPI (free tier, sleeps when idle)
            </text>
            <rect x={330} y={356} width={830} height={100} rx={18} className="arch-frame" />
            <text x={342} y={450} className="arch-frame-label">
              Supabase · PostgreSQL + pgvector
            </text>

            {/* Connectors: base line + flowing dots; the hop being travelled lights up */}
            {EDGES.map((e) => {
              const active = activeEdges.has(e.id) && filter === "all";
              const used = usedEdges.has(e.id) && filter === "all";
              const reverse = active && activeEdges.get(e.id) === false;
              const d = toPath(e.pts);
              return (
                <g key={e.id} data-arch-edge={e.id} data-state={active ? "active" : used ? "visited" : "idle"}>
                  <path
                    d={d}
                    className={`arch-edge ${e.ret ? "arch-edge-return" : ""}`}
                    style={active || used ? { stroke: hopColor, strokeOpacity: active ? 1 : 0.45 } : undefined}
                    strokeWidth={active ? 2.25 : 1.4}
                    markerEnd="url(#arch-arrow)"
                    markerStart={e.both ? "url(#arch-arrow)" : undefined}
                  />
                  <path
                    d={d}
                    className={`arch-flow ${active ? "arch-flow-active" : ""} ${reverse ? "arch-flow-reverse" : ""}`}
                    style={active ? { stroke: hopColor } : undefined}
                  />
                  <text x={e.lx} y={e.ly} className="arch-edge-label" textAnchor={e.mid ? "middle" : "start"}>
                    {e.label}
                  </text>
                </g>
              );
            })}

            {/* Boxes */}
            {NODES.map((n) => {
              const state = stateOf(n);
              const active = state === "active";
              const stroke = active && filter === "all" ? hopColor : KIND_STROKE[n.kind];
              const awaiting = active && filter === "all" && current.pause && n.id === current.node;
              return (
                <g
                  key={n.id}
                  data-arch-node={n.id}
                  data-state={state}
                  className={`arch-node arch-node-${state}`}
                  style={{ transformOrigin: `${n.x + n.w / 2}px ${n.y + NODE_H / 2}px` }}
                >
                  <rect
                    x={n.x}
                    y={n.y}
                    width={n.w}
                    height={NODE_H}
                    rx={12}
                    className={`arch-box arch-box-${n.kind}`}
                    style={{ stroke }}
                    strokeWidth={active ? 2.5 : n.kind === "plain" ? 1 : 1.75}
                  />
                  <text x={n.x + n.w / 2} y={n.y + 27} textAnchor="middle" className="arch-title">
                    {n.title}
                  </text>
                  <text x={n.x + n.w / 2} y={n.y + 46} textAnchor="middle" className="arch-sub">
                    {n.sub}
                  </text>
                  {awaiting && (
                    <g className="arch-await">
                      <rect x={n.x + n.w / 2 - 66} y={n.y - 30} width={132} height={20} rx={10} fill="var(--arch-human)" />
                      <text x={n.x + n.w / 2} y={n.y - 16} textAnchor="middle" className="arch-await-text">
                        ⏸ Awaiting human
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* The request packet */}
            {filter === "all" && (
              <g ref={packetRef} data-testid="arch-packet" className="arch-packet" style={{ color: hopColor }} transform={`translate(${topEdge(tour[0].node).join(" ")})`}>
                <circle r={13} className="arch-packet-halo" />
                <circle r={6.5} className="arch-packet-core" />
              </g>
            )}
          </svg>
        </div>

        {/* Step caption + progress */}
        <div className="mt-4 rounded-2xl bg-white/80 p-3 ring-1 ring-slate-200/80 backdrop-blur">
          <div className="flex gap-1" aria-hidden>
            {tour.map((t, i) => (
              <span
                key={i}
                className="h-1 flex-1 rounded-full bg-slate-200 transition-colors duration-300"
                style={i <= hop ? { backgroundColor: AUTHORITY_COLOR[t.authority] } : undefined}
              />
            ))}
          </div>
          <p aria-live="polite" data-testid="arch-caption" className="mt-2 text-sm text-slate-800">
            <span className="mr-2 font-mono text-xs text-slate-500">
              {hop + 1}/{tour.length}
            </span>
            <span aria-hidden className="mr-1 font-semibold" style={{ color: hopColor }}>
              {AUTHORITY[current.authority].mark}
            </span>
            {current.caption}
          </p>
        </div>

        {/* Legend */}
        <figcaption id="architecture-caption" className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-slate-600">
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="arch-swatch" style={{ borderColor: "var(--arch-model)" }} /> uses a language model
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="arch-swatch" style={{ borderColor: "var(--arch-authority)" }} /> holds authority
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="arch-swatch" style={{ borderColor: "var(--arch-knowledge)" }} /> policy knowledge (RAG)
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="arch-swatch" style={{ borderColor: "var(--arch-human)" }} /> human decision
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-px w-5 bg-slate-500" /> request flow
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="inline-block w-5 border-t border-dashed border-slate-500" /> answer back
          </span>
        </figcaption>
      </div>
    </figure>
  );
}
