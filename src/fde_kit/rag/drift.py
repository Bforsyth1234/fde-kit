"""Drift: is the evidence a recorded run saw still what the database holds now?

Replay answers "what did the agent see then"; drift answers "is that still true now". It reads
the Sources on a run's `tool` rows (src/fde_kit/record.py) and looks each one up again, inside
one `store.session(tenant)`, so RLS bounds the lookups exactly as it bounded the run. It
reports; it never re-answers, re-searches or re-runs anything.

    rec = RunRecord.load("runs/r-3f9a1c2e7b4d.jsonl")
    report = check_drift(rec, store)          # tenant from the run's header
    report.drifted, report.counts()           # True, {"unchanged": 4, "superseded": 1}

Sources are numbered and deduplicated by `source_id` the way the run's `CitationLedger` did, so
source #3 here is `[3]` in the run's answer.

What each tool's Source is checked against (see rag/tool.py for what each one emits):

- `search_documents` (source_id = chunk id, content_hash = the chunk's hash). A re-ingest
  deletes the old version's chunks, so the chunk is judged through its document: document
  deleted -> `gone`; document superseded -> `superseded` (old and current version/hash/doc_id);
  document current -> the chunk must still exist with the same hash.
- `fetch_document` (span `chars:a-b`, content_hash = the document's hash). Document deleted ->
  `gone`; superseded -> `superseded`; hash differs on the same row -> `value_changed` (a body
  edited in place, which `ingest` never does).
- `query_table` (span `fact:<fact_id>`, content_hash = the document's hash). Fact deleted ->
  `gone`; value differs -> `value_changed`; its document superseded -> `superseded`; now in a
  conflict group when the run saw it without one -> `new_conflict`.
- Any other tool, or a Source with no doc_id: `unchecked`, with the reason. Never an error.

What the record does and does not know about a fact. The Source stores the fact id, the
document and its hash, not the value. The value comes from the tool row's raw content: the
row line `[#i] <cells>` under the column header line, read back only when the SQL selected
`value_num` / `value_text` (and `unit`, if selected, is compared too). If it selected neither,
the value at record time is unknown and only existence, the document and the conflict state are
checked; the detail says so. The conflict state at record time is the ` conflict=true` suffix
`query_table` puts on every fact row in a conflict group (whatever columns were selected), so
`new_conflict` is reliable; a fact that already conflicted and whose group has since grown is
not detectable, because the record does not store group membership.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from ..record import RunRecord
from ..tools import Source
from .store import Fact, TenantSession, TenantStore
from .tool import ALL_TOOLS, UNVERIFIED_NOTE

Status = Literal["unchanged", "superseded", "gone", "value_changed", "new_conflict", "unchecked"]

STATUSES: tuple[Status, ...] = (
    "unchanged",
    "superseded",
    "gone",
    "value_changed",
    "new_conflict",
    "unchecked",
)
DRIFT_STATUSES: frozenset[str] = frozenset({"superseded", "gone", "value_changed", "new_conflict"})
"""Statuses that mean the run's evidence is no longer what the database holds."""


class TenantError(ValueError):
    """No tenant to check under, or a tenant that is not the run's."""


@dataclass
class SourceDrift:
    """One deduplicated Source of a run and what the database says about it now."""

    number: int
    """The run-wide citation number: `[number]` in the run's answer."""
    source_id: str
    tool: str
    title: str
    uri: str
    status: Status
    detail: str
    doc_id: str | None = None
    span: str | None = None
    then_hash: str | None = None
    now_hash: str | None = None
    then_version: int | None = None
    now_version: int | None = None
    now_doc_id: str | None = None
    then_value: str | None = None
    now_value: str | None = None
    then_conflict: bool | None = None
    now_conflict: bool | None = None
    flags: list[str] = field(default_factory=list)
    """Secondary findings, e.g. `new_conflict` on a fact whose status is `value_changed`."""

    @property
    def drifted(self) -> bool:
        return self.status in DRIFT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    run_id: str
    tenant_id: str
    items: list[SourceDrift]

    @property
    def drifted(self) -> bool:
        return any(i.drifted for i in self.items)

    def counts(self) -> dict[str, int]:
        """Status -> count, in `STATUSES` order, zero counts left out."""
        out = {s: sum(1 for i in self.items if i.status == s) for s in STATUSES}
        return {s: n for s, n in out.items() if n}

    def summary(self) -> str:
        n = len(self.items)
        parts = ", ".join(f"{c} {s}" for s, c in self.counts().items()) or "nothing to check"
        verdict = "drift" if self.drifted else "no drift"
        return f"{n} source{'s' if n != 1 else ''}: {parts} -> {verdict}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "drifted": self.drifted,
            "counts": self.counts(),
            "sources": [i.to_dict() for i in self.items],
        }


# --------------------------------------------------------------------------- entry points


def resolve_tenant(record: RunRecord, tenant: str | None = None) -> str:
    """`tenant` or the run header's `tenant_id`. Raises `TenantError` when there is neither, or
    when `tenant` is not the tenant the run was recorded for."""
    recorded = record.header.get("tenant_id")
    if tenant and recorded and tenant != recorded:
        raise TenantError(
            f"run {record.run_id} was recorded for tenant {recorded!r}, not {tenant!r}. Drift "
            "checks run under the run's own tenant: under row-level security a session only "
            "sees its own tenant's documents and facts, so checking under another tenant "
            "would report every source as gone."
        )
    resolved = tenant or recorded
    if not resolved:
        raise TenantError(
            f"run {record.run_id} has no tenant_id in its header; pass --tenant to say which "
            "tenant's database to check it against"
        )
    return str(resolved)


def check_drift(record: RunRecord, store: TenantStore, tenant: str | None = None) -> DriftReport:
    """Check every Source the run saw against the database, in one session for the tenant."""
    tid = resolve_tenant(record, tenant)
    seen = collect_sources(record)
    with store.session(tid) as s:
        items = [_check(s, tid, n, src, row) for n, src, row in seen]
    return DriftReport(run_id=record.run_id, tenant_id=tid, items=items)


def collect_sources(record: RunRecord) -> list[tuple[int, Source, dict[str, Any]]]:
    """(run-wide number, Source, the first tool row that returned it), deduplicated by
    `source_id` in first-seen order, which is how the run's CitationLedger numbered them."""
    out: list[tuple[int, Source, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in record.tool_rows():
        for d in row.get("sources") or []:
            src = Source.from_dict(d)
            if src.source_id in seen:
                continue
            seen.add(src.source_id)
            out.append((len(out) + 1, src, row))
    return out


# --------------------------------------------------------------------------- per source


def _check(s: TenantSession, tid: str, n: int, src: Source, row: dict[str, Any]) -> SourceDrift:
    tool = str(row.get("name") or "")
    item = SourceDrift(
        number=n,
        source_id=src.source_id,
        tool=tool,
        title=src.title,
        uri=src.uri,
        status="unchecked",
        detail="",
        doc_id=src.doc_id,
        span=src.span,
        then_hash=src.content_hash,
    )
    if tool not in ALL_TOOLS:
        item.detail = f"{tool or 'unknown tool'} is not a database tool; nothing to look up"
        return item
    if src.tenant_id and src.tenant_id != tid:
        item.detail = f"recorded for tenant {src.tenant_id!r}, not visible to {tid!r}"
        return item
    span = src.span or ""
    if span.startswith("fact:"):
        return _check_fact(s, item, span[len("fact:") :], src, row)
    if not src.doc_id:
        item.detail = "no doc_id on the source; nothing to look up"
        return item
    if span.startswith("chars:"):
        return _check_slice(s, item, src)
    if not span:
        return _check_chunk(s, item, src)
    item.detail = f"unrecognised span {span!r}"
    return item


@dataclass
class _Doc:
    doc_id: str
    source: str
    version: int
    content_hash: str
    superseded_by: str | None


def _doc(s: TenantSession, doc_id: str | None) -> _Doc | None:
    if not doc_id:
        return None
    try:
        key = str(uuid.UUID(str(doc_id)))
    except ValueError:
        return None
    r = s.conn.execute(
        s.sql(
            "SELECT doc_id::text, source, version, content_hash, superseded_by::text "
            "FROM @docs WHERE doc_id = %s::uuid"
        ),
        (key,),
    ).fetchone()
    return _Doc(r[0], r[1], int(r[2]), r[3], r[4]) if r else None


def _current(s: TenantSession, doc: _Doc) -> _Doc | None:
    """Follow `superseded_by` to the current version. If the chain is broken (a newer version
    was deleted), fall back to the current row for the same source, as `ingest` finds it."""
    cur, seen = doc, {doc.doc_id}
    while cur.superseded_by:
        nxt = _doc(s, cur.superseded_by)
        if nxt is None or nxt.doc_id in seen:
            r = s.conn.execute(
                s.sql(
                    "SELECT doc_id::text, source, version, content_hash, superseded_by::text "
                    "FROM @docs WHERE source = %s AND superseded_by IS NULL "
                    "ORDER BY version DESC LIMIT 1"
                ),
                (doc.source,),
            ).fetchone()
            return _Doc(r[0], r[1], int(r[2]), r[3], r[4]) if r else None
        seen.add(nxt.doc_id)
        cur = nxt
    return cur


def _short(h: str | None) -> str:
    return (h or "?")[:12]


def _superseded(s: TenantSession, item: SourceDrift, doc: _Doc, then_hash: str) -> SourceDrift:
    cur = _current(s, doc)
    item.status = "superseded"
    item.then_version, item.then_hash = doc.version, then_hash
    if cur is None:
        item.detail = f"v{doc.version} was superseded and no current version remains"
        return item
    item.now_version, item.now_hash, item.now_doc_id = cur.version, cur.content_hash, cur.doc_id
    item.detail = (
        f"re-ingested: v{doc.version} {_short(then_hash)} -> v{cur.version} "
        f"{_short(cur.content_hash)} (doc {cur.doc_id})"
    )
    if cur.content_hash == then_hash:
        item.detail += "; the current text is identical to the cited version"
    return item


def _check_chunk(s: TenantSession, item: SourceDrift, src: Source) -> SourceDrift:
    doc = _doc(s, src.doc_id)
    if doc is None:
        item.status, item.detail = "gone", f"document {src.doc_id} no longer exists"
        return item
    if doc.superseded_by:
        # The chunk's own hash is the chunk's; report the document's hashes, then and now.
        return _superseded(s, item, doc, doc.content_hash)
    item.then_version = item.now_version = doc.version
    try:
        key = str(uuid.UUID(src.source_id))
    except ValueError:
        key = None
    r = (
        s.conn.execute(
            s.sql("SELECT content_hash FROM @chunks WHERE chunk_id = %s::uuid AND doc_id = %s"),
            (key, doc.doc_id),
        ).fetchone()
        if key
        else None
    )
    if r is None:
        item.status = "gone"
        item.detail = f"passage no longer exists, though document v{doc.version} is current"
        return item
    item.now_hash = r[0]
    if src.content_hash and r[0] != src.content_hash:
        item.status = "value_changed"
        item.detail = f"passage text changed in place: {_short(src.content_hash)} -> {_short(r[0])}"
        return item
    item.status, item.detail = "unchanged", f"document v{doc.version} is current"
    return item


def _check_slice(s: TenantSession, item: SourceDrift, src: Source) -> SourceDrift:
    doc = _doc(s, src.doc_id)
    if doc is None:
        item.status, item.detail = "gone", f"document {src.doc_id} no longer exists"
        return item
    if src.content_hash and doc.content_hash != src.content_hash:
        item.status = "value_changed"
        item.then_version = item.now_version = doc.version
        item.now_hash = doc.content_hash
        item.detail = (
            f"document body changed in place: {_short(src.content_hash)} -> "
            f"{_short(doc.content_hash)}"
        )
        return item
    if doc.superseded_by:
        return _superseded(s, item, doc, src.content_hash or doc.content_hash)
    item.then_version = item.now_version = doc.version
    item.now_hash = doc.content_hash
    item.status, item.detail = "unchanged", f"document v{doc.version} is current"
    return item


# --------------------------------------------------------------------------- facts


@dataclass
class _RecordedFact:
    columns: dict[str, str]
    """Selected column -> rendered cell, as the run's query_table output showed it."""
    conflict: bool


def recorded_fact(row: dict[str, Any], fact_id: str) -> _RecordedFact | None:
    """What a `query_table` tool row showed for `fact_id`, read back from its raw content.

    The content is a header line of column names joined by " | ", then one line per row; a
    cited row starts with its local marker `[#i] `, may end with ` (source: ...)` and then
    ` conflict=true` (format_query_result). The fact's line is the marked line whose `fact_id`
    cell is this id. Returns None when the line is not found or its cells don't line up with
    the header (a cell containing " | " or "(source: "); the caller then treats the recorded
    value and conflict state as unknown."""
    lines = str(row.get("content") or "").splitlines()
    if not lines:
        return None
    cols = lines[0].split(" | ")
    if "fact_id" not in cols:
        return None
    for line in lines[1:]:
        if not line.startswith("[#"):
            continue
        body = line.split("] ", 1)[1] if "] " in line else ""
        conflict = body.endswith(" conflict=true")
        if conflict:
            body = body[: -len(" conflict=true")]
        if body.endswith(UNVERIFIED_NOTE):
            continue
        cut = body.rfind(" (source: ")
        if cut >= 0 and body.endswith(")"):
            body = body[:cut]
        cells = body.split(" | ")
        if len(cells) != len(cols):
            continue
        mapped = dict(zip(cols, cells, strict=True))
        if mapped.get("fact_id") == fact_id:
            return _RecordedFact(mapped, conflict)
    return None


def _render(v: Any) -> str:
    return "" if v is None else str(v)


def _same_value(then: str, now: float | str) -> bool:
    if isinstance(now, str):
        return then == now
    try:
        return Decimal(then) == Decimal(str(now))
    except InvalidOperation:
        return False


def _check_fact(
    s: TenantSession, item: SourceDrift, fact_id: str, src: Source, row: dict[str, Any]
) -> SourceDrift:
    # What the run saw, as far as the record shows it (see `recorded_fact`).
    rec = recorded_fact(row, fact_id)
    if rec is not None:
        item.then_conflict = rec.conflict
        item.then_value = rec.columns.get("value_num") or rec.columns.get("value_text") or None

    fact: Fact | None = s.facts_by_id([fact_id]).get(fact_id)
    if fact is None:
        item.status, item.detail = "gone", f"fact {fact_id} no longer exists"
        return item
    item.now_value = _render(fact.value)
    item.now_conflict = fact.conflict_group_id is not None
    item.now_doc_id = fact.doc_id
    item.now_hash = fact.doc_hash

    # Every finding, in priority order; the first is the status, the rest become flags.
    found: list[tuple[Status, str]] = []
    notes: list[str] = []

    if src.doc_id and fact.doc_id is None:
        found.append(("gone", f"fact remains but its document {src.doc_id} was deleted"))

    if rec is None:
        notes.append("fact row not found in the recorded output: value and conflict unknown")
    else:
        then_val = item.then_value
        changes: list[str] = []
        if then_val is None:
            notes.append("value not selected by the recorded SQL, so not compared")
        elif not _same_value(then_val, fact.value):
            changes.append(f"value {then_val} -> {item.now_value}")
        if "unit" in rec.columns and rec.columns["unit"] != (fact.unit or ""):
            changes.append(f"unit {rec.columns['unit'] or '-'} -> {fact.unit or '-'}")
        if changes:
            found.append(("value_changed", "; ".join(changes)))

    doc = _doc(s, src.doc_id) if src.doc_id and fact.doc_id else None
    if doc is not None:
        item.then_version = item.now_version = doc.version
        if doc.superseded_by:
            cur = _current(s, doc)
            if cur is not None:
                item.now_version, item.now_doc_id = cur.version, cur.doc_id
                item.now_hash = cur.content_hash
                found.append(
                    (
                        "superseded",
                        f"its document was re-ingested (v{doc.version} -> v{cur.version}); "
                        f"the fact still points at v{doc.version}",
                    )
                )
            else:
                found.append(("superseded", f"its document v{doc.version} was superseded"))

    if rec is not None and not rec.conflict and item.now_conflict:
        found.append(
            ("new_conflict", "now in a conflict group: another source reports a different value")
        )
    elif rec is not None and rec.conflict and not item.now_conflict:
        notes.append("no longer in a conflict group")

    if found:
        item.status = found[0][0]
        item.flags = [st for st, _ in found[1:]]
        parts = [msg for _, msg in found] + notes
    else:
        item.status = "unchanged"
        parts = [f"fact still {item.now_value}" if item.then_value else "fact exists", *notes]
    item.detail = "; ".join(parts)
    return item
