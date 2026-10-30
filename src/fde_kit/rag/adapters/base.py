"""The adapter contract: an external source becomes documents and facts with provenance.

An adapter yields `AdapterBatch`es (one per company, one per directory, ...); `ingest` loads
them through a `TenantSession`, which does everything else (versioning by content hash,
chunking, embedding, conflict groups, RLS).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

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
    facts_replaced: int = 0
    """Facts deleted because their document was superseded (`on_new_version="replace"`)."""


def ingest(
    session: TenantSession,
    adapter: Adapter,
    *,
    max_chars: int = 900,
    on_new_version: Literal["keep", "replace"] = "keep",
) -> IngestReport:
    """For each batch: upsert entities, ingest_versioned every document (idempotent by hash),
    one extraction job per call, add_facts with doc_id filled from document_source,
    set_dictionary. A fact whose document_source was not in the batch is an error. Re-running
    on unchanged input writes no documents; facts are only added for documents whose ingest
    changed (so a re-run does not duplicate facts).

    The extraction job is created lazily, on the first batch that has facts to write, so a
    re-run on unchanged input writes nothing at all. Its kind, model and prompt hash come from
    the adapter's optional `job_kind`, `job_model` and `job_prompt_sha256` attributes, with
    kind=f"adapter:{adapter.name}" and "" as the defaults.

    `on_new_version` says what happens to the facts of a document a new version supersedes:
    "keep" (the default) leaves them, so the new version's facts conflict with them where they
    differ; EDGAR wants that, since a filing is immutable and a restatement is another
    document. "replace" deletes them before the batch's facts are added, so a re-export
    replaces its own facts (they still conflict with other sources); the count is
    `facts_replaced`."""
    if on_new_version not in ("keep", "replace"):
        raise ValueError(f"on_new_version must be 'keep' or 'replace', got {on_new_version!r}")
    job_kind = getattr(adapter, "job_kind", f"adapter:{adapter.name}")
    job_model = getattr(adapter, "job_model", "")
    job_prompt_sha256 = getattr(adapter, "job_prompt_sha256", "")
    job_id = ""
    n_docs = n_changed = n_facts = n_replaced = 0
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
                if on_new_version == "replace" and outcome.previous_doc_id:
                    n_replaced += session.delete_facts_for_document(outcome.previous_doc_id)
        rows = [
            dataclasses.replace(f.row, doc_id=changed[f.document_source])
            for f in batch.facts
            if f.document_source in changed
        ]
        if rows:
            if not job_id:
                job_id = session.start_extraction_job(
                    job_kind, model=job_model, prompt_sha256=job_prompt_sha256
                )
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
        facts_replaced=n_replaced,
    )
