"""The agent-facing retrieval tools, bound to one tenant per request.

The tenant comes from the authenticated request (`tenant_id` here), never from the model. No
tool's schema has a tenant, namespace or filters argument, so there is no argument a prompt
injection could fill in to search somewhere else. Extra arguments the model invents are dropped
by the input model before the function runs; tests/test_rag_isolation.py sends one anyway.

Three tools, selected with `tools=`:

- `search_documents(query, k)`: hybrid search, passages numbered `[#i]` (PR 1).
- `fetch_document(doc_id, char_start, char_end)`: a slice of one stored document body.
- `query_table(sql)`: one read-only SELECT over the tenant's fact tables; the field dictionary
  is appended to the tool description when the registry is built.

Build a fresh registry per request:

    registry = tenant_search_registry(store, tenant_id=request.tenant)
    Agent(registry).run(request.question)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..tools import Source, ToolRegistry, ToolResult

# TODO(integrator): replace with `from .store import DictEntry, QueryResult, SqlRejected`
from ._contract import DictEntry, QueryResult, SqlRejected
from .schema import validate_tenant_id
from .store import Hit, TenantStore

MAX_K = 10

ALL_TOOLS: tuple[str, ...] = ("search_documents", "fetch_document", "query_table")

TABLES_PARAGRAPH = (
    "Tables: rag_facts(entity_id, field, period_start, period_end, value_num, value_text, "
    "unit, doc_id, chunk_id, conflict_group_id), rag_entities(entity_id, kind, "
    "canonical_name, aliases), rag_documents(doc_id, source, title, version). Join facts to "
    "entities on entity_id. Fields:"
)

TRUNCATED_NOTE = "... (truncated at {n} rows; add a WHERE or LIMIT)"


def tenant_search_registry(
    store: TenantStore,
    tenant_id: str,
    registry: ToolRegistry | None = None,
    *,
    cached: bool = True,
    tools: Sequence[str] = ALL_TOOLS,
) -> ToolRegistry:
    validate_tenant_id(tenant_id)
    unknown = set(tools) - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"unknown tools {sorted(unknown)}; choose from {list(ALL_TOOLS)}")
    if registry is not None and registry.tenant_id not in (None, tenant_id):
        raise ValueError(f"registry already belongs to tenant {registry.tenant_id!r}")
    # Never mutate a caller's shared base registry (where tenant_id is None) in place: doing so
    # would bind that shared instance and its `search_documents` tool to this request's tenant.
    if registry is None:
        out = ToolRegistry(tenant_id=tenant_id)
    elif registry.tenant_id is None:
        out = ToolRegistry(_tools=dict(registry._tools), tenant_id=tenant_id)
    else:
        out = registry

    if "search_documents" in tools:

        @out.tool(name="search_documents")
        def search_documents(query: str, k: int = 5) -> ToolResult:
            """Search this customer's documents (leases, policies, addenda) and return the most
            relevant passages, each numbered like [1], [2]. Use it for any question about terms,
            amounts, dates or rules, and cite the passage number you relied on in your answer."""
            k = max(1, min(int(k), MAX_K))
            with store.session(tenant_id) as s:
                hits = s.cached_search(query, k=k).hits if cached else s.search(query, k=k)
            if not hits:
                return ToolResult("No matching passages.")
            return format_passages(hits, tenant_id=tenant_id)

    if "fetch_document" in tools:

        @out.tool(name="fetch_document")
        def fetch_document(doc_id: str, char_start: int = 0, char_end: int = 6000) -> ToolResult:
            """Read part of one document by its doc_id (from a search result). Returns the text
            and, if more remains, `next: <offset>`; call again with char_start=<offset>."""
            with store.session(tenant_id) as s:
                try:
                    d = s.fetch(doc_id, char_start=char_start, char_end=char_end)
                except KeyError:
                    return ToolResult("No document with that id.", is_error=True)
            return format_slice(d, tenant_id=tenant_id)

    if "query_table" in tools:

        @out.tool(name="query_table")
        def query_table(sql: str) -> ToolResult:
            """Run one read-only SQL SELECT over this customer's fact tables. <DICTIONARY> is
            appended to this description at registry build time (see below). Rows marked
            conflict=true disagree with another source: report both values and say they
            conflict."""
            with store.session(tenant_id) as s:
                try:
                    result = s.query(sql)
                except SqlRejected as exc:
                    return ToolResult(f"Query rejected: {exc.reason}", is_error=True)
            return format_query_result(result, tenant_id=tenant_id)

        with store.session(tenant_id) as s:
            entries = s.dictionary()
        spec = out.get("query_table")
        spec.description = query_table_description(spec.description, entries)

    return out


def query_table_description(base_doc: str, entries: Sequence[DictEntry]) -> str:
    """`base_doc` + the tables paragraph + one `- field (unit): definition` line per entry."""
    lines = "".join(f"- {e.field} ({e.unit}): {e.definition}\n" for e in entries)
    return base_doc + "\n\n" + TABLES_PARAGRAPH + "\n" + lines


def format_passages(hits: list[Hit], *, tenant_id: str | None = None) -> ToolResult:
    """Number the hits with local markers `[#i]` and attach one Source per hit.

    The agent loop rewrites `[#i]` to run-wide `[n]` so numbers never collide across tool
    calls; the model only ever sees `[n]`. The Source carries chunk id, document, source URI
    and the chunk's content hash, which is what `AgentResult.cited` and the `cites` eval
    check resolve against. Nothing about the citation depends on the model's prose.
    """
    parts = [
        f"[#{i}] {h.title or h.source} ({h.source})\n{h.content}" for i, h in enumerate(hits, 1)
    ]
    return ToolResult(
        "\n\n".join(parts), sources=tuple(h.to_source(tenant_id) for h in hits)
    )


def format_slice(d: Any, *, tenant_id: str | None = None) -> ToolResult:
    """One `[#1]` header line, the slice text, and a `next:` line while more remains.

    `d` is a §3 `DocSlice`. The Source's id is `doc_id:start-end` so two slices of one document
    are two distinct citations, each provable by `content_hash` + `span`.
    """
    head = (
        f"[#1] {d.title or d.source} ({d.source}) v{d.version} "
        f"chars {d.char_start}-{d.char_end} of {d.total_chars}"
    )
    content = f"{head}\n{d.text}"
    if d.next is not None:
        content += f"\n\nnext: {d.next}"
    source = Source(
        source_id=f"{d.doc_id}:{d.char_start}-{d.char_end}",
        uri=d.source,
        title=d.title,
        doc_id=d.doc_id,
        content_hash=d.content_hash,
        span=f"chars:{d.char_start}-{d.char_end}",
        tenant_id=tenant_id,
    )
    return ToolResult(content, sources=(source,))


def format_query_result(result: QueryResult, *, tenant_id: str | None = None) -> ToolResult:
    """Render a `QueryResult` as a compact table and derive Sources from its columns.

    The model's SQL decides which columns come back, so the Source for a row is built from
    whatever provenance columns are present, mirroring `Fact.to_source`:

    - `fact_id` (required for a row to get a `[#i]` marker and a Source): `source_id`, and
      `span="fact:<fact_id>"`.
    - `doc_source` or `source`: `uri`; otherwise `uri="fact:<fact_id>"`.
    - `entity` + `field`: `title="<entity> <field>"` (either alone is used on its own).
    - `doc_id`: `doc_id`.
    - `doc_hash` or `content_hash`: `content_hash`.

    Rows without a `fact_id` column (e.g. `SELECT count(*)`) are printed without a marker and
    produce no Source. A row whose `conflict_group_id` column is non-null gets ` conflict=true`.
    """
    cols = list(result.columns)
    idx = {c: i for i, c in enumerate(cols)}
    has_fact = "fact_id" in idx
    conflict_col = idx.get("conflict_group_id")

    def col(row: tuple, *names: str) -> Any:
        for n in names:
            if n in idx and row[idx[n]] is not None:
                return row[idx[n]]
        return None

    lines = [" | ".join(cols)]
    sources: list[Source] = []
    n = 0
    for row in result.rows:
        cells = " | ".join("" if v is None else str(v) for v in row)
        prefix = ""
        if has_fact and row[idx["fact_id"]] is not None:
            n += 1
            prefix = f"[#{n}] "
            fact_id = str(row[idx["fact_id"]])
            entity, field = col(row, "entity"), col(row, "field")
            title = " ".join(str(x) for x in (entity, field) if x is not None)
            doc_hash = col(row, "doc_hash", "content_hash")
            doc_id = col(row, "doc_id")
            uri = col(row, "doc_source", "source")
            sources.append(
                Source(
                    source_id=fact_id,
                    uri=str(uri) if uri is not None else f"fact:{fact_id}",
                    title=title,
                    doc_id=str(doc_id) if doc_id is not None else None,
                    content_hash=str(doc_hash) if doc_hash is not None else None,
                    span=f"fact:{fact_id}",
                    tenant_id=tenant_id,
                )
            )
        line = prefix + cells
        if conflict_col is not None and row[conflict_col] is not None:
            line += " conflict=true"
        lines.append(line)
    if result.truncated:
        lines.append(TRUNCATED_NOTE.format(n=len(result.rows)))
    return ToolResult("\n".join(lines), sources=tuple(sources))
