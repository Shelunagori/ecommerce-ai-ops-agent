"""Deterministic, section-aware Markdown chunker for policy documents: ``policy-section-v1``.

    Markdown body
      -> sections at ATX headings (# .. ######) outside fenced code blocks; each section
         keeps its heading path ("Refund Policy > Refund timing") and its body verbatim
      -> a section that fits ``max_chars`` is ONE chunk (the common case for short policies)
      -> an oversized section is packed from blank-line blocks (lists, tables, fenced code
         and paragraphs stay whole); a block that alone is too large is packed from
         sentences; a sentence that alone is too large is hard-cut. Only these split pieces
         may carry ``overlap_chars`` of the previous piece; overlap never crosses sections.

Pure function of (body, title, settings): the same input always yields the same chunks.
Changing the algorithm requires a new ``CHUNKER`` name; changing the settings changes
``chunking_hash``. Either way ingestion refuses to replace chunks of an existing document.
"""

import hashlib
import json
import re
from dataclasses import dataclass

CHUNKER = "policy-section-v1"
MAX_SECTION_CHARS = 300

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 0

    def __post_init__(self) -> None:
        if not 200 <= self.max_chars <= 4000:
            raise ValueError("max_chars must be between 200 and 4000")
        if not 0 <= self.overlap_chars <= 300 or self.overlap_chars >= self.max_chars // 2:
            raise ValueError("overlap_chars must be 0-300 and less than half of max_chars")

    @classmethod
    def from_settings(cls, settings=None) -> "ChunkingConfig":  # noqa: ANN001
        from app.core.config import get_settings  # noqa: PLC0415

        s = settings or get_settings()
        return cls(s.knowledge_chunk_max_chars, s.knowledge_chunk_overlap_chars)

    @property
    def chunking_hash(self) -> str:
        spec = {"chunker": CHUNKER, "max_chars": self.max_chars, "overlap": self.overlap_chars}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyChunk:
    index: int
    section: str
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def chunk_policy(body: str, title: str, config: ChunkingConfig) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for section, text in _sections(body, title):
        for piece in _split_section(text, config):
            chunks.append(PolicyChunk(len(chunks), section, piece))
    if not chunks:
        raise ValueError("document produced no chunks")
    return chunks


def _sections(body: str, title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(line.rstrip() for line in lines).strip("\n")
        if text.strip():
            path = " > ".join(h for _, h in stack) or title
            sections.append((path[:MAX_SECTION_CHARS], text))
        lines.clear()

    for line in body.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            stack = [(lvl, h) for lvl, h in stack if lvl < level]
            stack.append((level, heading.group(2).strip()))
        else:
            lines.append(line)
    flush()
    return sections


def _split_section(text: str, config: ChunkingConfig) -> list[str]:
    if len(text) <= config.max_chars:
        return [text]
    budget = config.max_chars - (config.overlap_chars + 1 if config.overlap_chars else 0)
    pieces = _pack(_blocks(text), budget, "\n\n")
    if config.overlap_chars == 0:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        tail = prev[-config.overlap_chars :]
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail
        out.append(f"{tail.strip()}\n{cur}" if tail.strip() else cur)
    return out


def _blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(units: list[str], budget: int, joiner: str) -> list[str]:
    """Greedy, order-preserving packing of units into pieces of at most ``budget`` chars."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > budget:
            if current:
                pieces.append(current)
                current = ""
            if joiner == "\n\n":
                pieces.extend(_pack(_SENTENCE_END.split(unit), budget, " "))
            else:  # a single sentence longer than the budget: hard cut
                pieces.extend(unit[i : i + budget] for i in range(0, len(unit), budget))
            continue
        candidate = f"{current}{joiner}{unit}" if current else unit
        if len(candidate) <= budget:
            current = candidate
        else:
            pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]
