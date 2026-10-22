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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import psycopg
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


def content_hash(text: str) -> str:
    """sha256 hex of a chunk's text. What `Source.content_hash` carries."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        }

    def sql(self, template: str) -> str:
        """Fill a template's @chunks / @docs / @cache with this tenant's partitions."""
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
        """Replace-or-insert one document and its chunks. Returns doc_id (a UUID string)."""
        chunks = chunk_text(text, title=title, max_chars=max_chars)
        vecs = self.embedder.embed_documents(chunks) if chunks else []
        row = self.conn.execute(
            self.sql("""
            INSERT INTO @docs (tenant_id, source, title, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tenant_id, source)
            DO UPDATE SET title = EXCLUDED.title, metadata = EXCLUDED.metadata
            RETURNING doc_id::text
            """),
            (self.tenant_id, source, title, Jsonb(metadata or {})),
        ).fetchone()
        assert row is not None
        doc_id = row[0]
        self.conn.execute(self.sql("DELETE FROM @chunks WHERE doc_id = %s"), (doc_id,))
        with self.conn.cursor() as cur:
            cur.executemany(
                self.sql(
                    "INSERT INTO @chunks (tenant_id, doc_id, ord, content, content_hash, "
                    "embedding) VALUES (%s, %s, %s, %s, %s, %s::vector)"
                ),
                [
                    (self.tenant_id, doc_id, i, c, content_hash(c), to_pgvector(v))
                    for i, (c, v) in enumerate(zip(chunks, vecs, strict=True))
                ],
            )
        # Anything cached for this tenant may now be stale.
        self.conn.execute(self.sql("DELETE FROM @cache"))
        return doc_id

    # ------------------------------------------------------------------ reads

    def count_chunks(self) -> int:
        row = self.conn.execute(self.sql("SELECT count(*) FROM @chunks")).fetchone()
        return int(row[0]) if row else 0

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
