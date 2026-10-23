"""The adapter contract: an external source becomes documents and facts with provenance.

An adapter yields `AdapterBatch`es (one per company, one per directory, ...); `ingest` loads
them through a `TenantSession`, which does everything else (versioning by content hash,
chunking, embedding, conflict groups, RLS).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..store import DictEntry, FactRow

if TYPE_CHECKING:
    from ..store import TenantSession


@dataclass(frozen=True)
class RawDocument:
    source: str
    """Stable URI, e.g. the sec.gov archive URL of the document."""
    title: str
    text: str
    """Already extracted plain text."""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RawFact:
    document_source: str
    """Which RawDocument (by source) this fact is cited to."""
    row: FactRow
    """doc_id left None; `ingest` fills it from document_source."""


@dataclass(frozen=True)
class RawEntity:
    kind: str
    canonical_name: str
    aliases: tuple[str, ...] = ()


@dataclass
class AdapterBatch:
    """One unit of ingest, e.g. one company."""

    entities: list[RawEntity]
    documents: list[RawDocument]
    facts: list[RawFact]
    dictionary: list[DictEntry] = field(default_factory=list)


class Adapter(Protocol):
    name: str
    """"edgar", "local_files", ..."""

    def batches(self) -> Iterator[AdapterBatch]: ...


@dataclass
class IngestReport:
    documents: int
    """Documents seen."""
    documents_changed: int
    """Documents written (new source or new content hash)."""
    facts: int
    """Fact rows written."""
    conflict_groups: int
    """Distinct conflict groups among the facts of the entities this run touched."""
    entities: int
    job_id: str
    """The extraction job the facts were written under; "" when no facts were written."""


def ingest(session: TenantSession, adapter: Adapter, *, max_chars: int = 900) -> IngestReport:
    """For each batch: upsert entities, ingest_versioned every document (idempotent by hash),
    one extraction job per call (kind=f"adapter:{adapter.name}"), add_facts with doc_id filled
    from document_source, set_dictionary. A fact whose document_source was not in the batch is
    an error. Re-running on unchanged input writes no documents; facts are only added for
    documents whose ingest changed (so a re-run does not duplicate facts).

    The extraction job is created lazily, on the first batch that has facts to write, so a
    re-run on unchanged input writes nothing at all."""
    job_id = ""
    n_docs = n_changed = n_facts = 0
    entity_names: list[str] = []
    for batch in adapter.batches():
        sources = {d.source for d in batch.documents}
        unknown = sorted({f.document_source for f in batch.facts} - sources)
        if unknown:
            raise ValueError(
                f"adapter {adapter.name!r}: facts cite documents not in their batch: {unknown}"
            )
        for e in batch.entities:
            session.upsert_entity(e.kind, e.canonical_name, e.aliases)
            if e.canonical_name not in entity_names:
                entity_names.append(e.canonical_name)
        changed: dict[str, str] = {}
        for d in batch.documents:
            outcome = session.ingest_versioned(
                d.source, d.text, title=d.title, metadata=dict(d.metadata), max_chars=max_chars
            )
            n_docs += 1
            if outcome.changed:
                n_changed += 1
                changed[d.source] = outcome.doc_id
        rows = [
            dataclasses.replace(f.row, doc_id=changed[f.document_source])
            for f in batch.facts
            if f.document_source in changed
        ]
        if rows:
            if not job_id:
                job_id = session.start_extraction_job(f"adapter:{adapter.name}")
            n_facts += len(session.add_facts(rows, extraction_job_id=job_id))
        if batch.dictionary:
            session.set_dictionary(batch.dictionary)
    groups = {
        f.conflict_group_id
        for name in entity_names
        for f in session.facts(entity=name)
        if f.conflict_group_id
    }
    return IngestReport(
        documents=n_docs,
        documents_changed=n_changed,
        facts=n_facts,
        conflict_groups=len(groups),
        entities=len(entity_names),
        job_id=job_id,
    )
