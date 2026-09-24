/**
 * Splits an assistant answer into text and citation segments.
 *
 * Citations are the backend's canonical tenant-relative URIs `policy://<key>/v<n>#chunk-<i>`
 * (usually in square brackets). Only citations the backend returned as VALIDATED
 * (`AgentResponse.citations`) get metadata; the backend rejects any answer with an
 * unretrieved citation, so an unknown one should not occur - it is shown as plain text.
 */
import type { PolicyCitation } from "./types";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; citation: string; source: PolicyCitation; index: number };

const CITATION = /\[?(policy:\/\/[a-z0-9]+(?:-[a-z0-9]+)*\/v[1-9][0-9]*#chunk-(?:0|[1-9][0-9]*))\]?/g;

export function segmentAnswer(answer: string, citations: PolicyCitation[]): Segment[] {
  const byUri = new Map(citations.map((c) => [c.citation, c]));
  const order = new Map<string, number>();
  const segments: Segment[] = [];
  let last = 0;
  for (const match of answer.matchAll(CITATION)) {
    const uri = match[1];
    const source = byUri.get(uri);
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: answer.slice(last, start) });
    if (source) {
      if (!order.has(uri)) order.set(uri, order.size + 1);
      segments.push({ kind: "citation", citation: uri, source, index: order.get(uri)! });
    } else {
      segments.push({ kind: "text", text: match[0] });
    }
    last = start + match[0].length;
  }
  if (last < answer.length) segments.push({ kind: "text", text: answer.slice(last) });
  return segments;
}

export function formatEffective(c: Pick<PolicyCitation, "effective_from" | "effective_to">): string {
  return c.effective_to ? `${c.effective_from} – ${c.effective_to}` : `since ${c.effective_from}`;
}
