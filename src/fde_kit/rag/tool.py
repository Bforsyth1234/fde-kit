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

from collections.abc import Mapping, Sequence
from typing import Any

from ..tools import Source, ToolRegistry, ToolResult
from .schema import validate_tenant_id
from .store import DictEntry, Fact, Hit, QueryResult, SqlRejected, TenantStore

MAX_K = 10

ALL_TOOLS: tuple[str, ...] = ("search_documents", "fetch_document", "query_table")

TABLES_PARAGRAPH = (
    "Tables: rag_facts(fact_id, entity_id, field, period_start, period_end, value_num, "
    "value_text, unit, doc_id, chunk_id, conflict_group_id), rag_entities(entity_id, kind, "
    "canonical_name, aliases), rag_documents(doc_id, source, title, version). Join facts to "
    "entities on entity_id; match names case-insensitively, e.g. e.canonical_name ILIKE "
    "'%pfizer%'. One plain SELECT (or WITH ... SELECT) only; system catalogs such as "
    "information_schema and pg_* are not available. Example:\n"
    "SELECT f.fact_id, e.canonical_name, f.field, f.period_start, f.period_end, f.value_num, "
    "f.unit FROM rag_facts f JOIN rag_entities e ON e.entity_id = f.entity_id "
    "WHERE e.canonical_name ILIKE '%pfizer%' AND f.field = 'revenue' ORDER BY f.period_end\n"
    "Fields:"
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
    # Never mutate the caller's registry: a shared base (tenant_id None) would end up bound to
    # this request's tenant. Copy its tools into a fresh registry instead, leaving out any
    # retrieval tools it already has (e.g. it is the output of an earlier call for this
    # tenant), so they are re-registered here rather than raising "tool already registered".
    out = ToolRegistry(tenant_id=tenant_id)
    if registry is not None:
        out._tools.update({n: spec for n, spec in registry._tools.items() if n not in ALL_TOOLS})

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
            """Run one read-only SQL SELECT over this customer's fact tables. Select f.fact_id
            so each row is numbered, citable, and labelled with the document it came from and
            whether it conflicts; you do not need to join documents yourself. Rows marked
            conflict=true disagree with another source: report every value and say they
            conflict. When conflicting rows were "reported as" different line items, say so:
            that is a reclassification, not a correction of the same figure."""
            # The tables paragraph, an example query and one line per dictionary field are
            # appended to this description when the registry is built; see
            # query_table_description().
            with store.session(tenant_id) as s:
                try:
                    result = s.query(sql)
                except SqlRejected as exc:
                    return ToolResult(f"Query rejected: {exc.reason}", is_error=True)
                # Provenance from the database, not from the columns the model chose: every
                # row with a fact_id gets its document's URL, title and hash, and its conflict
                # status. Same tenant session, so RLS still bounds what can be looked up.
                facts = s.facts_by_id(fact_ids_in(result))
            return format_query_result(result, tenant_id=tenant_id, facts=facts)

        with store.session(tenant_id) as s:
            entries = s.dictionary()
        spec = out.get("query_table")
        spec.description = query_table_description(spec.description, entries)

    return out


def query_table_description(base_doc: str, entries: Sequence[DictEntry]) -> str:
    """`base_doc` + the tables paragraph + one `- field (unit): definition` line per entry.
    A field with no single unit (late_fee is USD or percent_of_rent) gets no parentheses."""
    lines = "".join(
        f"- {e.field}{f' ({e.unit})' if e.unit else ''}: {e.definition}\n" for e in entries
    )
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


def reported_line_item(source_ref: str) -> str:
    """The line item a fact was reported under, from its `source_ref`.

    Adapters write `"<taxonomy>:<Tag>@<accession>"` (EDGAR: `us-gaap:ResearchAndDevelopmentExpense@
    0000882095-20-000006`); this returns the part before `@`. Empty for facts without one (the
    lease demo). Showing it lets the model see *why* two values conflict when the reason is a
    reclassification: Gilead's FY2019 R&D is $9,106M as ResearchAndDevelopmentExpense and
    $4,055M as ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost.
    """
    return source_ref.split("@", 1)[0] if source_ref else ""


def fact_ids_in(result: QueryResult) -> list[str]:
    """The non-null values of a `fact_id` column, if the result has one."""
    if "fact_id" not in result.columns:
        return []
    k = list(result.columns).index("fact_id")
    return [str(row[k]) for row in result.rows if row[k] is not None]


def format_query_result(
    result: QueryResult, *, tenant_id: str | None = None, facts: Mapping[str, Fact] | None = None
) -> ToolResult:
    """Render a `QueryResult` as a compact table with a `[#i]` and a Source per fact row.

    A row gets a marker and a Source only if the SQL selected `fact_id`. Its provenance comes
    from `facts` (looked up from the database by `query_table`), not from the columns the model
    happened to select:

    - Source = `Fact.to_source`: uri is the document's URL, title "<entity> <field> <period>",
      doc_id, the document's content hash, span `fact:<fact_id>`.
    - The line the model reads ends with ` (source: <document title>)`, or
      ` (source: <document title>; reported as <line item>)` when the fact has a `source_ref`
      (the XBRL tag for EDGAR facts), and with
      ` conflict=true` when the fact is in a conflict group, even if neither column was
      selected. This is what lets the model say which filing each number came from without a
      second query.

    A fact_id that is not in `facts` (not a real fact id, or called without a lookup) falls back
    to the columns: `doc_source`/`source` for uri (else `fact:<fact_id>`), `entity` + `field`
    for title, `doc_id`, `doc_hash`/`content_hash`, and `conflict_group_id` for the marker.
    Rows without a `fact_id` column (e.g. `SELECT count(*)`) get no marker and no Source.
    """
    facts = facts or {}
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
        line = " | ".join("" if v is None else str(v) for v in row)
        conflict = conflict_col is not None and row[conflict_col] is not None
        if has_fact and row[idx["fact_id"]] is not None:
            n += 1
            line = f"[#{n}] {line}"
            fact_id = str(row[idx["fact_id"]])
            fact = facts.get(fact_id)
            if fact is not None:
                sources.append(fact.to_source(tenant_id))
                label = fact.doc_title or fact.doc_source
                reported_as = reported_line_item(fact.source_ref)
                if label and reported_as:
                    line += f" (source: {label}; reported as {reported_as})"
                elif label:
                    line += f" (source: {label})"
                conflict = conflict or fact.conflict_group_id is not None
            else:
                entity, field = col(row, "entity"), col(row, "field")
                doc_hash = col(row, "doc_hash", "content_hash")
                doc_id = col(row, "doc_id")
                uri = col(row, "doc_source", "source")
                sources.append(
                    Source(
                        source_id=fact_id,
                        uri=str(uri) if uri is not None else f"fact:{fact_id}",
                        title=" ".join(str(x) for x in (entity, field) if x is not None),
                        doc_id=str(doc_id) if doc_id is not None else None,
                        content_hash=str(doc_hash) if doc_hash is not None else None,
                        span=f"fact:{fact_id}",
                        tenant_id=tenant_id,
                    )
                )
        if conflict:
            line += " conflict=true"
        lines.append(line)
    if result.truncated:
        lines.append(TRUNCATED_NOTE.format(n=len(result.rows)))
    return ToolResult("\n".join(lines), sources=tuple(sources))
