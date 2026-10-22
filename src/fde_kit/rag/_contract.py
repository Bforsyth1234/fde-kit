"""TEMPORARY: PR 3 §3 contract types, duplicated so `tool.py` imports before `store.py` lands.

The integrator (wave 3-G) replaces `from ._contract import ...` in tool.py with
`from .store import ...` and deletes this file. Field names and order match §3 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..tools import Source


@dataclass
class DocSlice:
    doc_id: str
    source: str
    title: str
    version: int
    content_hash: str
    superseded_by: str | None
    char_start: int
    char_end: int
    text: str
    total_chars: int
    next: int | None


@dataclass
class DictEntry:
    field: str
    definition: str
    unit: str = ""
    allowed_values: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class Fact:
    fact_id: str
    entity_id: str
    entity: str
    field: str
    period_start: date | None
    period_end: date | None
    value: float | str
    unit: str
    doc_id: str | None
    chunk_id: str | None
    doc_source: str | None
    doc_hash: str | None
    conflict_group_id: str | None
    extraction_job_id: str | None

    def to_source(self, tenant_id: str | None = None) -> Source:
        return Source(
            source_id=self.fact_id,
            uri=self.doc_source or f"fact:{self.fact_id}",
            title=f"{self.entity} {self.field}",
            doc_id=self.doc_id,
            content_hash=self.doc_hash,
            span=f"fact:{self.fact_id}",
            tenant_id=tenant_id,
        )


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


class SqlRejected(ValueError):
    """query_table refused the statement before sending it: not a single SELECT, or it
    touches something off-limits. `.reason` is a short human string."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
