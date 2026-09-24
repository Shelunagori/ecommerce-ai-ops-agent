"""Policy source loader: discover, parse and validate synthetic policy Markdown files.

Rules (never relaxed by file content):
* Only ``<root>/<tenant-slug>/<document_key>.v<version>.md`` files are read (exactly two
  levels deep). Every path is resolved and must stay inside ``root`` (symlinks included).
* Front matter is a ``---`` block parsed with ``yaml.safe_load`` (no code execution) and
  validated by a strict model (unknown keys rejected). Metadata cannot name a file path.
* The ``tenant`` field must be a known tenant slug AND match the directory; the file name
  must match ``document_key`` and ``version``.
* The body is data: it is never executed or interpreted beyond Markdown headings.
"""

import hashlib
import json
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
_SLUG = re.compile(SLUG_PATTERN)
_FILE_NAME = re.compile(r"^(?P<key>[a-z0-9]+(?:-[a-z0-9]+)*)\.v(?P<version>[1-9][0-9]{0,3})\.md$")
_FRONT_MATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n(?P<body>.*)\Z", re.S)
MAX_SOURCE_BYTES = 64_000

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"


class PolicySourceError(ValueError):
    """A policy source is invalid. ``source`` is the path relative to the root (safe)."""

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"{source}: {reason}")


class PolicyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant: str = Field(pattern=SLUG_PATTERN, max_length=63)
    document_key: str = Field(pattern=SLUG_PATTERN, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    document_type: Literal["policy"]
    version: int = Field(ge=1, le=9999)
    effective_from: date
    effective_to: date | None = None

    @model_validator(mode="after")
    def _range(self) -> "PolicyMetadata":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from")
        return self


@dataclass(frozen=True)
class PolicySource:
    metadata: PolicyMetadata
    body: str  # Markdown after the front matter, line endings normalised to "\n"
    source_name: str  # relative to the policy root, e.g. "northstar-commerce/refund.v1.md"
    immutable_content_hash: str  # see ``compute_immutable_content_hash``

    @property
    def identity(self) -> tuple[str, str, int]:
        m = self.metadata
        return (m.tenant, m.document_key, m.version)


def normalise(raw: str) -> str:
    """Only line endings are normalised; any other change is a content change."""
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def compute_immutable_content_hash(metadata: "PolicyMetadata", body: str) -> str:
    """sha256 of the IMMUTABLE policy payload - NOT a hash of the whole source file.

    Covers tenant, document_key, title, document_type, version, effective_from and the body
    (line endings normalised). ``effective_to`` is the single, deliberately excluded field:
    it is the one-time retirement field (``null -> date`` once, when a successor version is
    published; never changed afterwards), which ingestion checks separately."""
    immutable = metadata.model_dump(mode="json", exclude={"effective_to"})
    payload = json.dumps({"metadata": immutable, "body": body}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_policy(raw: str, source_name: str) -> PolicySource:
    text = normalise(raw)
    match = _FRONT_MATTER.match(text)
    if not match:
        raise PolicySourceError(source_name, "missing or malformed front matter")
    try:
        meta: Any = yaml.safe_load(match.group("meta"))
    except (yaml.YAMLError, ValueError):  # ValueError: e.g. an impossible date literal
        raise PolicySourceError(source_name, "front matter is not valid YAML") from None
    if not isinstance(meta, dict):
        raise PolicySourceError(source_name, "front matter must be a mapping")
    try:
        metadata = PolicyMetadata.model_validate(meta)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "metadata" for e in exc.errors()})
        raise PolicySourceError(source_name, f"invalid metadata: {', '.join(fields)}") from None
    body = match.group("body")
    if not body.strip():
        raise PolicySourceError(source_name, "empty document body")
    return PolicySource(metadata, body, source_name, compute_immutable_content_hash(metadata, body))


def load_policy_file(root: Path, path: Path, known_tenants: Collection[str]) -> PolicySource:
    root = root.resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        raise PolicySourceError(path.name, "file is outside the policy directory") from None
    source_name = rel.as_posix()
    if len(rel.parts) != 2 or path.is_symlink():
        raise PolicySourceError(source_name, "unexpected location (expected <tenant>/<file>.md)")
    tenant_dir, file_name = rel.parts
    name = _FILE_NAME.match(file_name)
    if not _SLUG.fullmatch(tenant_dir) or not name:
        raise PolicySourceError(source_name, "file name must be <document_key>.v<version>.md")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise PolicySourceError(source_name, "file too large")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise PolicySourceError(source_name, "file is not valid UTF-8") from None
    source = parse_policy(raw, source_name)
    m = source.metadata
    if m.tenant not in known_tenants:
        raise PolicySourceError(source_name, "unknown tenant")
    if m.tenant != tenant_dir:
        raise PolicySourceError(source_name, "tenant does not match its directory")
    if (m.document_key, m.version) != (name["key"], int(name["version"])):
        raise PolicySourceError(source_name, "file name does not match document_key/version")
    return source


def discover_policy_files(root: Path) -> list[Path]:
    """``<root>/<slug>/*.md`` only, sorted; anything resolving outside ``root`` is refused."""
    root = root.resolve()
    if not root.is_dir():
        raise PolicySourceError(str(root.name), "policy directory does not exist")
    files: list[Path] = []
    for tenant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(tenant_dir.glob("*.md")):
            files.append(path)
    return files


def load_policy_sources(
    root: Path, known_tenants: Collection[str], paths: Iterable[Path] | None = None
) -> list[PolicySource]:
    """Load and validate every policy file; duplicate identities are rejected."""
    sources = [
        load_policy_file(root, p, known_tenants)
        for p in (paths if paths is not None else discover_policy_files(root))
    ]
    seen: dict[tuple[str, str, int], str] = {}
    for s in sources:
        if s.identity in seen:
            raise PolicySourceError(s.source_name, f"duplicate of {seen[s.identity]}")
        seen[s.identity] = s.source_name
    validate_version_ranges([(s.metadata, s.source_name) for s in sources])
    return sorted(sources, key=lambda s: s.identity)


def validate_version_ranges(items: Iterable[tuple[Any, str]]) -> None:
    """Per (tenant, document_key): versions ordered by version must have strictly later
    ``effective_from`` and must not overlap (half-open ranges). Only the last may be open.
    Items expose tenant/document_key/version/effective_from/effective_to."""
    groups: dict[tuple[str, str], list[tuple[Any, str]]] = {}
    for item, name in items:
        groups.setdefault((item.tenant, item.document_key), []).append((item, name))
    for group in groups.values():
        ordered = sorted(group, key=lambda x: x[0].version)
        for (prev, _), (cur, name) in zip(ordered, ordered[1:], strict=False):
            if cur.effective_from <= prev.effective_from:
                raise PolicySourceError(name, "version must start after the previous version")
            if prev.effective_to is None or prev.effective_to > cur.effective_from:
                raise PolicySourceError(name, "effective range overlaps the previous version")
