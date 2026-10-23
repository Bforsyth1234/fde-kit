"""Tenant-scoped retrieval over Postgres + pgvector.

    store = TenantStore(app_url, HashEmbedder())
    with store.session("harbor") as s:
        s.ingest("leases/4b.md", text, title="Lease: Unit 4B")
        hits = s.search("what is the late fee?", k=5)

Every database call happens inside `session(tenant_id)`, which opens a transaction and sets
`app.tenant_id` with `set_config(..., is_local => true)`. Transaction-local is the important
part. A session-level SET on a pooled connection survives the request that made it, and the
next request to borrow that connection, for a different tenant, runs as the previous one. That
leak is reproduced in tests/test_rag_isolation.py.

No query in this module filters on tenant_id. That is on purpose: the policy does it, and code
that also filters would hide a broken policy from the tests.

Search modes:
- vector:  cosine distance on the HNSW index.
- keyword: BM25 over Postgres full-text lexemes, with per-tenant corpus statistics.
- hybrid:  both, fused by relative score (default) or reciprocal rank. The default mode.

Tables are partitioned by tenant (see schema.py), and the policy's condition prunes every other
tenant's partition at run time, so each search walks only this tenant's own HNSW, GIN and B-tree
indexes. On a shared index, HNSW returns its `ef_search` nearest rows across *all* tenants and
the policy filters afterwards: a small tenant can get zero results, and with pgvector's
iterative scan turned on to fix that, the scan's length (so its latency) depends on how much
*other* tenants have written about your query. tests/test_rag_isolation.py reproduces both on a
deliberately shared table. Iterative scan stays on here because metadata filters still
post-filter within a tenant.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

import psycopg
from psycopg import sql as pgsql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ..tools import Source
from .chunk import chunk_text
from .embed import Embedder, to_pgvector
from .schema import (
    TENANT_SETTING,
    InsecureConnection,
    check_posture,
    partition_name,
    validate_tenant_id,
)
from .sqlguard import SqlRejected, check_select

__all__ = [
    "DictEntry",
    "DocSlice",
    "Fact",
    "FactRow",
    "Hit",
    "IngestOutcome",
    "QueryResult",
    "SearchResult",
    "SqlRejected",
    "TenantSession",
    "TenantStore",
    "chunk_spans",
    "content_hash",
    "fuse",
]

SearchMode = Literal["vector", "keyword", "hybrid"]
Fusion = Literal["relative", "rrf"]
RRF_K = 60


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    content: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    content_hash: str = ""
    """sha256 of `content` as stored. Empty only for rows ingested before the column existed."""

    def to_source(self, tenant_id: str | None = None) -> Source:
        return Source(
            source_id=self.chunk_id,
            uri=self.source,
            title=self.title,
            doc_id=self.doc_id,
            content_hash=self.content_hash or None,
            tenant_id=tenant_id,
        )


@dataclass
class SearchResult:
    hits: list[Hit]
    cached: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestOutcome:
    doc_id: str
    version: int
    changed: bool
    """False = identical hash, nothing written."""
    content_hash: str


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
    """Char offset to continue from, None at end."""


@dataclass
class DictEntry:
    field: str
    definition: str
    unit: str = ""
    allowed_values: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class FactRow:
    """INPUT shape for `add_facts`."""

    entity: str
    """Canonical name or alias; resolved to entity_id (error if unknown)."""
    field: str
    value: float | int | str
    """A number goes to value_num, a str to value_text."""
    unit: str = ""
    period_start: date | None = None
    period_end: date | None = None
    doc_id: str | None = None
    chunk_id: str | None = None


@dataclass
class Fact:
    """OUTPUT shape of `facts`."""

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
    """More than max_rows existed."""
    elapsed_ms: float


def content_hash(text: str) -> str:
    """sha256 hex of a chunk's text. What `Source.content_hash` carries."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_spans(
    text: str, chunks: list[str], title: str = ""
) -> list[tuple[int | None, int | None]]:
    """(char_start, char_end) of each chunk's body inside `text`, or (None, None) when the body
    can't be found verbatim from the previous chunk's end (chunk_text re-joins paragraphs with
    a fixed separator, so a document with unusual blank lines may not locate). Title-prefixed
    chunks are located by their body, after the title line."""
    prefix = f"{title}\n\n" if title else ""
    spans: list[tuple[int | None, int | None]] = []
    pos = 0
    for c in chunks:
        body = c[len(prefix) :] if prefix and c.startswith(prefix) else c
        at = text.find(body, pos) if body else -1
        if at < 0:
            spans.append((None, None))
        else:
            spans.append((at, at + len(body)))
            pos = at + len(body)
    return spans


def _plain(v: Any) -> Any:
    """Row values as plain Python: UUIDs as str, NUMERIC as int or float."""
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    return v


def fuse(
    vec: list[Hit], kw: list[Hit], k: int, method: Fusion = "relative", alpha: float = 0.5
) -> list[Hit]:
    """Merge two ranked lists. Each input Hit carries its own retriever's score and rank."""
    merged: dict[int, Hit] = {}
    for h in vec:
        merged[h.chunk_id] = Hit(**{**asdict(h), "score": 0.0, "keyword_rank": None})
    for h in kw:
        m = merged.get(h.chunk_id)
        if m is None:
            merged[h.chunk_id] = Hit(**{**asdict(h), "score": 0.0, "vector_rank": None})
        else:
            m.keyword_rank = h.keyword_rank

    if method == "rrf":
        for h in merged.values():
            h.score = sum(1.0 / (RRF_K + r) for r in (h.vector_rank, h.keyword_rank) if r)
    elif method == "relative":
        for weight, hits in ((alpha, vec), (1 - alpha, kw)):
            if not hits:
                continue
            hi, lo = max(h.score for h in hits), min(h.score for h in hits)
            for h in hits:
                norm = (h.score - lo) / (hi - lo) if hi > lo else 1.0
                merged[h.chunk_id].score += weight * norm
    else:
        raise ValueError(f"unknown fusion method: {method}")
    return sorted(merged.values(), key=lambda h: (-h.score, h.chunk_id))[:k]


class TenantStore:
    def __init__(
        self,
        url: str,
        embedder: Embedder,
        *,
        pool_size: int = 4,
        ef_search: int = 40,
        iterative_scan: bool = True,
        allow_insecure: bool = False,
    ):
        """`allow_insecure=True` skips the posture check. It exists so the tests can show what
        a superuser connection does to isolation; nothing else should pass it."""
        self.embedder = embedder
        self.ef_search = ef_search
        self.iterative_scan = iterative_scan
        self.pool = ConnectionPool(url, min_size=1, max_size=pool_size, open=True)
        if not allow_insecure:
            with self.pool.connection() as conn:
                problems = check_posture(conn)
            if problems:
                self.pool.close()
                raise InsecureConnection(problems)

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> TenantStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def session(self, tenant_id: str) -> Iterator[TenantSession]:
        validate_tenant_id(tenant_id)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config(%s, %s, true)", (TENANT_SETTING, tenant_id))
            # User-settable, so a pooled connection could carry a changed value; pin it.
            conn.execute("SET LOCAL log_parameter_max_length_on_error = 0")
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            if self.iterative_scan:
                conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            yield TenantSession(conn, tenant_id, self.embedder)


class TenantSession:
    """Every statement goes to this tenant's own partitions, named explicitly, so Postgres plans
    against one tenant's tables instead of all of them. Planning through the parent tables
    considers every partition before pruning, which made planning time grow with the number
    of tenants (0.2 ms at 8, 5.6 ms at 208): slow, and a way to count the customers. Each
    partition carries the same forced policy as its parent, so a session pointed at the wrong
    tenant's partition would still see nothing."""

    def __init__(self, conn: psycopg.Connection, tenant_id: str, embedder: Embedder):
        self.conn = conn
        self.tenant_id = tenant_id
        self.embedder = embedder
        # Partition names are derived from a hash and are always [a-z0-9_]; quoting is safe.
        self._tables = {
            "@chunks": f'"{partition_name("rag_chunks", tenant_id)}"',
            "@docs": f'"{partition_name("rag_documents", tenant_id)}"',
            "@cache": f'"{partition_name("rag_cache", tenant_id)}"',
            "@entities": f'"{partition_name("rag_entities", tenant_id)}"',
            "@facts": f'"{partition_name("rag_facts", tenant_id)}"',
            "@dictionary": f'"{partition_name("rag_dictionary", tenant_id)}"',
            "@jobs": f'"{partition_name("rag_extraction_jobs", tenant_id)}"',
        }
        self.last_ingest: IngestOutcome | None = None

    def sql(self, template: str) -> str:
        """Fill a template's @chunks / @docs / @cache / @entities / @facts / @dictionary /
        @jobs with this tenant's partitions."""
        for token, name in self._tables.items():
            template = template.replace(token, name)
        return template

    # ------------------------------------------------------------------ writes

    def ingest(
        self,
        source: str,
        text: str,
        *,
        title: str = "",
        metadata: dict[str, Any] | None = None,
        max_chars: int = 900,
    ) -> str:
        """Ingest one document (versioned; see `ingest_versioned`). Returns the current doc_id;
        the full outcome is on `self.last_ingest`."""
        outcome = self.ingest_versioned(
            source, text, title=title, metadata=metadata, max_chars=max_chars
        )
        return outcome.doc_id

    def ingest_versioned(
        self,
        source: str,
        text: str,
        *,
        title: str = "",
        metadata: dict[str, Any] | None = None,
        max_chars: int = 900,
    ) -> IngestOutcome:
        """Idempotent by sha256 of `text`.

        - same source, same hash: nothing written; the current doc_id with `changed=False`.
        - same source, new hash: a new row at version max+1 with the body, the previous current
          row's `superseded_by` pointed at it, the previous version's chunks deleted (search
          only ever sees current chunks) and this tenant's cache cleared. The old row stays as
          provenance, so an old citation's doc_id + content_hash still resolves.
        """
        digest = content_hash(text)
        current = self.conn.execute(
            self.sql(
                "SELECT doc_id::text, version, content_hash FROM @docs "
                "WHERE source = %s AND superseded_by IS NULL ORDER BY version DESC LIMIT 1"
            ),
            (source,),
        ).fetchone()
        if current is not None and current[2] == digest:
            outcome = IngestOutcome(current[0], int(current[1]), False, digest)
            self.last_ingest = outcome
            return outcome

        chunks = chunk_text(text, title=title, max_chars=max_chars)
        vecs = self.embedder.embed_documents(chunks) if chunks else []
        row = self.conn.execute(
            self.sql("""
            INSERT INTO @docs (tenant_id, source, title, metadata, content_hash, version, body)
            VALUES (%(t)s, %(s)s, %(title)s, %(m)s, %(h)s,
                    (SELECT COALESCE(max(version), 0) + 1 FROM @docs WHERE source = %(s)s),
                    %(body)s)
            RETURNING doc_id::text, version
            """),
            {
                "t": self.tenant_id,
                "s": source,
                "title": title,
                "m": Jsonb(metadata or {}),
                "h": digest,
                "body": text,
            },
        ).fetchone()
        assert row is not None
        doc_id, version = row[0], int(row[1])
        if current is not None:
            self.conn.execute(
                self.sql("UPDATE @docs SET superseded_by = %s WHERE doc_id = %s"),
                (doc_id, current[0]),
            )
            self.conn.execute(self.sql("DELETE FROM @chunks WHERE doc_id = %s"), (current[0],))
        spans = chunk_spans(text, chunks, title)
        with self.conn.cursor() as cur:
            cur.executemany(
                self.sql(
                    "INSERT INTO @chunks (tenant_id, doc_id, ord, content, content_hash, "
                    "char_start, char_end, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)"
                ),
                [
                    (self.tenant_id, doc_id, i, c, content_hash(c), cs, ce, to_pgvector(v))
                    for i, (c, v, (cs, ce)) in enumerate(zip(chunks, vecs, spans, strict=True))
                ],
            )
        # Anything cached for this tenant may now be stale.
        self.conn.execute(self.sql("DELETE FROM @cache"))
        outcome = IngestOutcome(doc_id, version, True, digest)
        self.last_ingest = outcome
        return outcome

    # ------------------------------------------------------------------ entities / facts

    def upsert_entity(self, kind: str, canonical_name: str, aliases: Iterable[str] = ()) -> str:
        """Insert or update (by kind + canonical name; aliases are replaced). Returns entity_id."""
        row = self.conn.execute(
            self.sql("""
            INSERT INTO @entities (tenant_id, kind, canonical_name, aliases)
            VALUES (%s, %s, %s, %s::text[])
            ON CONFLICT (tenant_id, kind, canonical_name) DO UPDATE SET aliases = EXCLUDED.aliases
            RETURNING entity_id::text
            """),
            (self.tenant_id, kind, canonical_name, list(dict.fromkeys(aliases))),
        ).fetchone()
        assert row is not None
        return row[0]

    def resolve_entity(self, name: str) -> str | None:
        """entity_id for a canonical name or alias, case-insensitive; None when unknown. A
        canonical match wins over an alias match."""
        row = self.conn.execute(
            self.sql("""
            SELECT entity_id::text FROM @entities
            WHERE lower(canonical_name) = lower(%(n)s::text)
               OR EXISTS (SELECT 1 FROM unnest(aliases) AS al(a)
                          WHERE lower(al.a) = lower(%(n)s::text))
            ORDER BY (lower(canonical_name) = lower(%(n)s::text)) DESC, canonical_name
            LIMIT 1
            """),
            {"n": name},
        ).fetchone()
        return row[0] if row else None

    def start_extraction_job(self, kind: str, *, model: str = "", prompt_sha256: str = "") -> str:
        row = self.conn.execute(
            self.sql(
                "INSERT INTO @jobs (tenant_id, kind, model, prompt_sha256) "
                "VALUES (%s, %s, %s, %s) RETURNING extraction_job_id::text"
            ),
            (self.tenant_id, kind, model, prompt_sha256),
        ).fetchone()
        assert row is not None
        return row[0]

    def add_facts(self, rows: list[FactRow], *, extraction_job_id: str | None = None) -> list[str]:
        """Insert facts; returns their fact_ids in order.

        Conflict rule: existing facts with the same (entity_id, field, period_start,
        period_end) and a *different* value share one conflict_group_id with the new row (the
        existing group's id if any of them has one, else a new uuid). An existing fact with the
        same value is left alone and the new row is still inserted: two sources agreeing is
        evidence, not a duplicate, and not a conflict."""
        out: list[str] = []
        for r in rows:
            entity_id = self.resolve_entity(r.entity)
            if entity_id is None:
                raise ValueError(f"unknown entity {r.entity!r} for tenant {self.tenant_id}")
            if isinstance(r.value, bool) or not isinstance(r.value, int | float | str):
                raise ValueError(f"fact value must be a number or a string, got {r.value!r}")
            num = r.value if isinstance(r.value, int | float) else None
            txt = r.value if isinstance(r.value, str) else None
            existing = self.conn.execute(
                self.sql("""
                SELECT fact_id::text, conflict_group_id::text,
                       (value_num IS NOT DISTINCT FROM %(num)s::numeric
                        AND value_text IS NOT DISTINCT FROM %(txt)s::text) AS same
                FROM @facts
                WHERE entity_id = %(e)s::uuid AND field = %(f)s::text
                  AND period_start IS NOT DISTINCT FROM %(ps)s::date
                  AND period_end IS NOT DISTINCT FROM %(pe)s::date
                """),
                {
                    "e": entity_id,
                    "f": r.field,
                    "ps": r.period_start,
                    "pe": r.period_end,
                    "num": num,
                    "txt": txt,
                },
            ).fetchall()
            differing = [row for row in existing if not row[2]]
            group: str | None = None
            if differing:
                group = next((g for _, g, _ in differing if g), None) or str(uuid.uuid4())
                self.conn.execute(
                    self.sql(
                        "UPDATE @facts SET conflict_group_id = %s::uuid "
                        "WHERE fact_id = ANY(%s::uuid[]) "
                        "AND conflict_group_id IS DISTINCT FROM %s::uuid"
                    ),
                    (group, [f for f, _, _ in differing], group),
                )
            row = self.conn.execute(
                self.sql("""
                INSERT INTO @facts (tenant_id, entity_id, field, period_start, period_end,
                                    value_num, value_text, unit, doc_id, chunk_id,
                                    conflict_group_id, extraction_job_id)
                VALUES (%s, %s::uuid, %s, %s::date, %s::date, %s::numeric, %s::text, %s,
                        %s::uuid, %s::uuid, %s::uuid, %s::uuid)
                RETURNING fact_id::text
                """),
                (
                    self.tenant_id,
                    entity_id,
                    r.field,
                    r.period_start,
                    r.period_end,
                    num,
                    txt,
                    r.unit,
                    r.doc_id,
                    r.chunk_id,
                    group,
                    extraction_job_id,
                ),
            ).fetchone()
            assert row is not None
            out.append(row[0])
        return out

    def facts(self, entity: str | None = None, field: str | None = None) -> list[Fact]:
        """This tenant's facts, optionally for one entity (canonical name or alias) and/or one
        field. An unknown entity name yields an empty list."""
        entity_id: str | None = None
        if entity is not None:
            entity_id = self.resolve_entity(entity)
            if entity_id is None:
                return []
        rows = self.conn.execute(
            self.sql("""
            SELECT f.fact_id::text, f.entity_id::text, e.canonical_name, f.field,
                   f.period_start, f.period_end, f.value_num, f.value_text, f.unit,
                   f.doc_id::text, f.chunk_id::text, d.source, d.content_hash,
                   f.conflict_group_id::text, f.extraction_job_id::text
            FROM @facts f
            JOIN @entities e ON e.entity_id = f.entity_id
            LEFT JOIN @docs d ON d.doc_id = f.doc_id
            WHERE (%(e)s::uuid IS NULL OR f.entity_id = %(e)s::uuid)
              AND (%(f)s::text IS NULL OR f.field = %(f)s::text)
            ORDER BY e.canonical_name, f.field, f.period_start NULLS FIRST, f.created_at, f.fact_id
            """),
            {"e": entity_id, "f": field},
        ).fetchall()
        return [
            Fact(
                fact_id=r[0],
                entity_id=r[1],
                entity=r[2],
                field=r[3],
                period_start=r[4],
                period_end=r[5],
                value=_plain(r[6]) if r[6] is not None else r[7],
                unit=r[8],
                doc_id=r[9],
                chunk_id=r[10],
                doc_source=r[11],
                doc_hash=r[12] or None,
                conflict_group_id=r[13],
                extraction_job_id=r[14],
            )
            for r in rows
        ]

    def set_dictionary(self, entries: Iterable[DictEntry]) -> None:
        """Upsert by field."""
        params = [
            (self.tenant_id, e.field, e.definition, e.unit, list(e.allowed_values), e.notes)
            for e in entries
        ]
        if not params:
            return
        with self.conn.cursor() as cur:
            cur.executemany(
                self.sql("""
                INSERT INTO @dictionary (tenant_id, field, definition, unit, allowed_values, notes)
                VALUES (%s, %s, %s, %s, %s::text[], %s)
                ON CONFLICT (tenant_id, field) DO UPDATE SET
                    definition = EXCLUDED.definition, unit = EXCLUDED.unit,
                    allowed_values = EXCLUDED.allowed_values, notes = EXCLUDED.notes
                """),
                params,
            )

    def dictionary(self) -> list[DictEntry]:
        rows = self.conn.execute(
            self.sql(
                "SELECT field, definition, unit, allowed_values, notes FROM @dictionary "
                "ORDER BY field"
            )
        ).fetchall()
        return [DictEntry(r[0], r[1], r[2], tuple(r[3]), r[4]) for r in rows]

    # ------------------------------------------------------------------ reads

    def count_chunks(self) -> int:
        row = self.conn.execute(self.sql("SELECT count(*) FROM @chunks")).fetchone()
        return int(row[0]) if row else 0

    def fetch(self, doc_id: str, char_start: int = 0, char_end: int = 6000) -> DocSlice:
        """A slice of one document's stored body; the range is clamped to the body.

        Raises KeyError for an unknown doc_id. Under RLS another tenant's doc_id is unknown, so
        the error is identical for "no such document" and "not yours"."""
        try:
            key = str(uuid.UUID(str(doc_id)))
        except ValueError:
            raise KeyError(doc_id) from None
        row = self.conn.execute(
            self.sql(
                "SELECT source, title, version, content_hash, superseded_by::text, body "
                "FROM @docs WHERE doc_id = %s::uuid"
            ),
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        source, title, version, digest, superseded_by, body = row
        total = len(body)
        start = max(0, min(int(char_start), total))
        end = max(start, min(int(char_end), total))
        return DocSlice(
            doc_id=key,
            source=source,
            title=title,
            version=int(version),
            content_hash=digest,
            superseded_by=superseded_by,
            char_start=start,
            char_end=end,
            text=body[start:end],
            total_chars=total,
            next=end if end < total else None,
        )

    def query(self, sql: str, *, max_rows: int = 200, timeout_ms: int = 5000) -> QueryResult:
        """Run one read-only SELECT under this tenant's RLS session.

        The statement first passes `check_select` (a keyword filter), then runs after
        `SET LOCAL transaction_read_only = on` and a statement timeout, wrapped in
        `SELECT * FROM (<sql>) AS q LIMIT max_rows + 1`. The model writes against the parent
        tables (`rag_facts`, `rag_entities`, `rag_documents`, `rag_dictionary`); the policy
        applies through them. Any psycopg error comes back as `SqlRejected` carrying the first
        line of the server's message.

        The statement runs inside a savepoint, so a rejected one doesn't poison the session's
        transaction. The read-only switch is made *outside* the savepoint: Postgres saves and
        restores transaction_read_only around every subtransaction, so set inside it would be
        undone by RELEASE. Set at the top level it holds, and Postgres allows switching a
        transaction to read-only but never back, so once query() has run the rest of this
        session is read-only: use a fresh session to write. The statement timeout is set
        inside the savepoint and put back afterwards.
        """
        cleaned = check_select(sql)
        max_rows = max(1, int(max_rows))
        prev = self.conn.execute("SHOW statement_timeout").fetchone()
        t0 = time.perf_counter()
        try:
            self.conn.execute("SET LOCAL transaction_read_only = on")
            with self.conn.transaction():  # SAVEPOINT; an error rolls it back
                self.conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                # No parameters on purpose: the model's SQL may contain '%' (LIKE patterns)
                # and must not be parsed for placeholders.
                cur = self.conn.execute(f"SELECT * FROM ({cleaned}) AS q LIMIT {max_rows + 1}")
                columns = [d.name for d in cur.description or ()]
                rows = cur.fetchall()
        except psycopg.Error as exc:
            primary = getattr(exc.diag, "message_primary", None) if hasattr(exc, "diag") else None
            reason = primary or (str(exc).strip().splitlines() or ["query failed"])[0]
            raise SqlRejected(reason) from exc
        elapsed = (time.perf_counter() - t0) * 1000
        self.conn.execute(
            pgsql.SQL("SET LOCAL statement_timeout = {}").format(
                pgsql.Literal(prev[0] if prev else 0)
            )
        )
        return QueryResult(
            columns=columns,
            rows=[tuple(_plain(v) for v in r) for r in rows[:max_rows]],
            truncated=len(rows) > max_rows,
            elapsed_ms=elapsed,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: SearchMode = "hybrid",
        filters: dict[str, Any] | None = None,
        fusion: Fusion = "relative",
        alpha: float = 0.5,
        candidates: int = 50,
    ) -> list[Hit]:
        """`filters` is matched against document metadata with JSONB containment. It can only
        narrow results: it is AND-ed inside the policy, never a substitute for it.

        Hybrid runs both retrievers for `candidates` rows each and fuses in Python. `fusion`:
        - "relative": min-max normalise each list's scores to [0, 1], then
          alpha * vector + (1 - alpha) * keyword. The default; see README for why.
        - "rrf": reciprocal rank fusion, sum of 1 / (60 + rank).
        """
        f = Jsonb(filters or {})
        if mode == "keyword":
            return self._keyword(query, f, k)
        qvec = to_pgvector(self.embedder.embed_query(query))
        if mode == "vector":
            return self._vector(qvec, f, k)
        if mode != "hybrid":
            raise ValueError(f"unknown search mode: {mode}")
        n = max(candidates, k)
        return fuse(self._vector(qvec, f, n), self._keyword(query, f, n), k, fusion, alpha)

    def _vector(self, qvec: str, f: Jsonb, n: int) -> list[Hit]:
        rows = self.conn.execute(
            self.sql(_VECTOR_SQL), {"q": qvec, "f": f, "n": n, "k": n}
        ).fetchall()
        return [Hit(*r) for r in rows]

    def _keyword(self, query: str, f: Jsonb, n: int) -> list[Hit]:
        rows = self.conn.execute(
            self.sql(_KEYWORD_SQL), {"t": query, "f": f, "n": n, "k": n}
        ).fetchall()
        return [Hit(*r) for r in rows]

    def cached_search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: SearchMode = "hybrid",
        filters: dict[str, Any] | None = None,
        max_distance: float = 0.08,
    ) -> SearchResult:
        """Semantic cache in front of `search`. A paraphrase within `max_distance` (cosine) of a
        previous query reuses its result.

        The cache table is under the same policy as the chunks, so the lookup can only ever
        match this tenant's previous queries. A process-wide cache keyed on the query embedding
        is the classic multi-tenant leak: two customers asking the same question share an
        answer. tests/test_rag_isolation.py reproduces it with exactly that cache.
        """
        params = {"k": k, "mode": mode, "filters": filters or {}}
        qvec = to_pgvector(self.embedder.embed_query(query))
        row = self.conn.execute(
            self.sql(CACHE_LOOKUP_SQL), {"p": Jsonb(params), "q": qvec, "d": max_distance}
        ).fetchone()
        if row is not None:
            return SearchResult([Hit(**h) for h in row[0]], cached=True, params=params)
        hits = self.search(query, k=k, mode=mode, filters=filters)
        key = hashlib.sha256(
            json.dumps([query.strip().lower(), params], sort_keys=True).encode()
        ).hexdigest()
        self.conn.execute(
            self.sql("""
            INSERT INTO @cache (tenant_id, query_hash, query, embedding, params, result)
            VALUES (%s, %s, %s, %s::vector, %s, %s)
            ON CONFLICT (tenant_id, query_hash) DO UPDATE SET result = EXCLUDED.result
            """),
            (
                self.tenant_id,
                key,
                query,
                qvec,
                Jsonb(params),
                Jsonb([asdict(h) for h in hits]),
            ),
        )
        return SearchResult(hits, cached=False, params=params)


CACHE_LOOKUP_SQL = """
SELECT result FROM @cache
WHERE params = %(p)s AND embedding <=> %(q)s::vector < %(d)s
ORDER BY embedding <=> %(q)s::vector LIMIT 1
"""

# Shared projection: (chunk_id, doc_id, source, title, content, score, vector_rank, keyword_rank)
#
# Candidate queries scan rag_chunks alone and ORDER BY the distance directly, so the HNSW index
# can drive the scan; ranks are computed on the candidates afterwards. A window function over
# the full table, or a join in the ordered scan, makes the planner fall back to sorting every
# row. The metadata filter is an EXISTS on the scanned row, which iterative scan can re-check.

_FILTER = (
    "(%(f)s = '{}'::jsonb OR EXISTS (SELECT 1 FROM @docs fd "
    "WHERE fd.doc_id = c.doc_id AND fd.metadata @> %(f)s))"
)

_VEC_CANDIDATES = f"""
    SELECT c.chunk_id, c.embedding <=> %(q)s::vector AS dist
    FROM @chunks c
    WHERE {_FILTER}
    ORDER BY c.embedding <=> %(q)s::vector
    LIMIT %(n)s
"""

# Keyword retrieval is BM25 (k1=1.2, b=0.75), computed in SQL. Postgres's own ts_rank has no
# IDF: a word in every chunk ("rent", "lease") counts as much as one in a single chunk (a lease
# reference number), which made keyword search useless on this corpus.
#
# The corpus statistics (N, average length, document frequency) come from queries over
# rag_chunks, and those queries run under the same RLS policy as everything else. So IDF is
# per tenant without any extra code. With global statistics, one tenant's documents would shift
# another tenant's scores, and a tenant watching its own rankings could infer which terms the
# others' documents contain. tests/test_rag_isolation.py checks that adding documents to one
# tenant leaves another tenant's scores bit-for-bit unchanged.
#
# The df subquery scans the tenant's chunks on every search. Fine at demo scale; a large
# deployment would maintain a per-tenant term-stats table instead.

_KW_CANDIDATES = f"""
    WITH ql AS (
        SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(t)s))
    ),
    stats AS (
        SELECT count(*)::float8 AS n, GREATEST(avg(n_tokens), 1)::float8 AS avgdl
        FROM @chunks
    ),
    df AS (
        SELECT u.lexeme, count(*)::float8 AS df
        FROM @chunks c, unnest(c.tsv) u
        WHERE u.lexeme IN (SELECT lexeme FROM ql)
        GROUP BY u.lexeme
    ),
    matches AS (
        SELECT c.chunk_id, c.n_tokens, u.lexeme, array_length(u.positions, 1)::float8 AS tf
        FROM @chunks c, unnest(c.tsv) u
        WHERE u.lexeme IN (SELECT lexeme FROM ql) AND {_FILTER}
    )
    SELECT m.chunk_id,
           sum(
               ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
               * m.tf * 2.2 / (m.tf + 1.2 * (0.25 + 0.75 * m.n_tokens / s.avgdl))
           ) AS rank
    FROM matches m JOIN df USING (lexeme), stats s
    GROUP BY m.chunk_id
    ORDER BY rank DESC, m.chunk_id
    LIMIT %(n)s
"""

_PROJECT = """
SELECT c.chunk_id::text, c.doc_id::text, d.source, d.title, c.content, {score}::float8, {vr}, {tr},
       c.content_hash
FROM {src}
JOIN @chunks c ON c.chunk_id = x.chunk_id
JOIN @docs d ON d.doc_id = c.doc_id
ORDER BY {order}
LIMIT %(k)s
"""

_VECTOR_SQL = _PROJECT.format(
    src=f"(SELECT chunk_id, dist, row_number() OVER (ORDER BY dist)::int AS r "
    f"FROM ({_VEC_CANDIDATES}) v) x",
    score="1 - x.dist",
    vr="x.r",
    tr="NULL::int",
    order="x.r",
)

_KEYWORD_SQL = _PROJECT.format(
    src=f"(SELECT chunk_id, rank, row_number() OVER (ORDER BY rank DESC, chunk_id)::int AS r "
    f"FROM ({_KW_CANDIDATES}) t) x",
    score="x.rank",
    vr="NULL::int",
    tr="x.r",
    order="x.r",
)
