"""The three tenant tools (PR 3 §4) against a fake store. No Postgres.

Covers: tool selection, the dictionary folded into `query_table`'s description, content
formats, Sources built from whatever columns the model's SQL returned, error paths, and the
no-tenant-argument property that the isolation suite relies on.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from fde_kit.rag.store import DictEntry, DocSlice, Fact, Hit, QueryResult, SearchResult, SqlRejected
from fde_kit.rag.tool import format_passages, tenant_search_registry
from fde_kit.tools import Source, ToolRegistry

# ------------------------------------------------------------------ fakes

HIT = Hit(
    chunk_id="chunk-1",
    doc_id="doc-1",
    source="harbor/leases/unit-4b.md",
    title="Unit 4B lease",
    content="Late fee is $75.",
    score=0.9,
    content_hash="h1",
)

DOC = DocSlice(
    doc_id="doc-1",
    source="harbor/leases/unit-4b.md",
    title="Unit 4B lease",
    version=2,
    content_hash="abc123",
    superseded_by=None,
    char_start=0,
    char_end=16,
    text="Late fee is $75.",
    total_chars=40,
    next=16,
)

DICT = [
    DictEntry(field="late_fee", definition="Fee charged after the grace period", unit="USD"),
    DictEntry(
        field="grace_days", definition="Days after the due date before a late fee", unit="days"
    ),
]


@dataclass
class FakeSession:
    hits: list[Hit] = field(default_factory=lambda: [HIT])
    docs: dict[str, DocSlice] = field(default_factory=lambda: {"doc-1": DOC})
    result: QueryResult | Exception = field(
        default_factory=lambda: QueryResult(columns=[], rows=[], truncated=False, elapsed_ms=0)
    )
    entries: list[DictEntry] = field(default_factory=lambda: list(DICT))
    calls: list[tuple] = field(default_factory=list)

    def search(self, query, *, k=5):
        self.calls.append(("search", query, k))
        return self.hits

    def cached_search(self, query, *, k=5):
        self.calls.append(("cached_search", query, k))
        return SearchResult(hits=self.hits, cached=True)

    def fetch(self, doc_id, char_start=0, char_end=6000):
        self.calls.append(("fetch", doc_id, char_start, char_end))
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return self.docs[doc_id]

    def query(self, sql, *, max_rows=200, timeout_ms=5000):
        self.calls.append(("query", sql))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def dictionary(self):
        self.calls.append(("dictionary",))
        return self.entries


class FakeStore:
    def __init__(self, session: FakeSession | None = None):
        self.s = session or FakeSession()
        self.tenants: list[str] = []

    @contextmanager
    def session(self, tenant_id):
        self.tenants.append(tenant_id)
        yield self.s


def registry(session: FakeSession | None = None, **kw) -> tuple[ToolRegistry, FakeStore]:
    store = FakeStore(session)
    return tenant_search_registry(store, "harbor", **kw), store


def props(reg: ToolRegistry, name: str) -> set[str]:
    return set(reg.get(name).to_api_schema()["input_schema"]["properties"])


# ------------------------------------------------------------------ registry shape


def test_registry_has_three_tools_by_default():
    reg, _ = registry()
    assert reg.names() == ["search_documents", "fetch_document", "query_table"]
    assert reg.tenant_id == "harbor"


def test_tools_selection_reproduces_search_only_registry():
    reg, store = registry(tools=("search_documents",))
    assert reg.names() == ["search_documents"]
    # No dictionary lookup happens when query_table is not registered.
    assert store.s.calls == []


def test_unknown_tool_name_is_rejected():
    with pytest.raises(ValueError):
        registry(tools=("search_documents", "query_everything"))


def test_no_tool_exposes_a_tenant_argument():
    reg, _ = registry()
    assert props(reg, "search_documents") == {"query", "k"}
    assert props(reg, "fetch_document") == {"doc_id", "char_start", "char_end"}
    assert props(reg, "query_table") == {"sql"}
    for schema in reg.api_schemas():
        keys = set(schema["input_schema"]["properties"])
        assert not keys & {"tenant_id", "tenant", "namespace", "filters"}


def test_injected_tenant_argument_is_dropped_by_every_tool():
    session = FakeSession(
        result=QueryResult(columns=["n"], rows=[(3,)], truncated=False, elapsed_ms=1.0)
    )
    reg, store = registry(session)
    extra = {"tenant_id": "summit", "tenant": "summit", "filters": {"tenant_id": "summit"}}
    out = reg.execute("query_table", {"sql": "SELECT count(*) AS n FROM rag_facts", **extra})
    assert not out.is_error and out.content == "n\n3"
    out = reg.execute("fetch_document", {"doc_id": "doc-1", **extra})
    assert not out.is_error
    out = reg.execute("search_documents", {"query": "late fee", **extra})
    assert not out.is_error
    # Every session was opened for the request's tenant, never the model's.
    assert set(store.tenants) == {"harbor"}


# ------------------------------------------------------------------ search_documents


def test_search_documents_is_unchanged_from_pr1():
    reg, store = registry()
    out = reg.execute("search_documents", {"query": "late fee", "k": 3})
    assert out == format_passages([HIT], tenant_id="harbor")
    assert out.content == "[#1] Unit 4B lease (harbor/leases/unit-4b.md)\nLate fee is $75."
    assert out.sources == (HIT.to_source("harbor"),)
    assert ("cached_search", "late fee", 3) in store.s.calls


def test_search_documents_uncached_and_empty():
    session = FakeSession(hits=[])
    reg, store = registry(session, cached=False)
    out = reg.execute("search_documents", {"query": "late fee", "k": 50})
    assert out.content == "No matching passages." and out.sources == ()
    assert ("search", "late fee", 10) in store.s.calls  # k clamped to MAX_K


# ------------------------------------------------------------------ fetch_document


def test_fetch_document_content_and_source():
    reg, store = registry()
    out = reg.execute("fetch_document", {"doc_id": "doc-1", "char_end": 16})
    assert not out.is_error
    assert out.content == (
        "[#1] Unit 4B lease (harbor/leases/unit-4b.md) v2 chars 0-16 of 40\n"
        "Late fee is $75.\n\n"
        "next: 16"
    )
    assert out.sources == (
        Source(
            source_id="doc-1:0-16",
            uri="harbor/leases/unit-4b.md",
            title="Unit 4B lease",
            doc_id="doc-1",
            content_hash="abc123",
            span="chars:0-16",
            tenant_id="harbor",
        ),
    )
    assert ("fetch", "doc-1", 0, 16) in store.s.calls


def test_fetch_document_omits_next_line_at_end():
    last = DocSlice(**{**DOC.__dict__, "char_start": 16, "char_end": 40, "next": None})
    reg, _ = registry(FakeSession(docs={"doc-1": last}))
    out = reg.execute("fetch_document", {"doc_id": "doc-1", "char_start": 16})
    assert "next:" not in out.content
    assert out.content.startswith(
        "[#1] Unit 4B lease (harbor/leases/unit-4b.md) v2 chars 16-40 of 40\n"
    )
    assert out.sources[0].source_id == "doc-1:16-40"


def test_fetch_document_unknown_id_is_an_error():
    reg, _ = registry()
    out = reg.execute("fetch_document", {"doc_id": "doc-from-summit"})
    assert out.is_error and out.content == "No document with that id."
    assert out.sources == ()


# ------------------------------------------------------------------ query_table description


def test_query_table_description_has_tables_paragraph_and_dictionary_lines():
    reg, store = registry()
    desc = reg.get("query_table").description
    assert desc.startswith("Run one read-only SQL SELECT over this customer's fact tables.")
    assert "\n\nTables: rag_facts(entity_id, field, period_start, period_end, value_num, " in desc
    assert "Join facts to entities on entity_id. Fields:\n" in desc
    assert "- late_fee (USD): Fee charged after the grace period\n" in desc
    assert "- grace_days (days): Days after the due date before a late fee\n" in desc
    assert desc.index("late_fee") < desc.index("grace_days")
    assert reg.api_schemas()[2]["description"] == desc
    # Built once per registry, from the tenant's session.
    assert store.s.calls.count(("dictionary",)) == 1
    assert store.tenants == ["harbor"]


# ------------------------------------------------------------------ query_table content


def fact_result(**overrides) -> QueryResult:
    base = dict(
        columns=[
            "fact_id",
            "entity",
            "field",
            "value_num",
            "doc_id",
            "doc_source",
            "doc_hash",
            "conflict_group_id",
        ],
        rows=[
            ("f1", "Harbor PM", "late_fee", 75, "doc-1", "harbor/leases/unit-4b.md", "h1", "cg1"),
            ("f2", "Harbor PM", "late_fee", 85, "doc-2", "harbor/addenda/late-fee.md", "h2", "cg1"),
            ("f3", "Harbor PM", "grace_days", 5, "doc-1", "harbor/leases/unit-4b.md", "h1", None),
        ],
        truncated=False,
        elapsed_ms=2.0,
    )
    return QueryResult(**{**base, **overrides})


def test_query_table_content_format():
    reg, _ = registry(FakeSession(result=fact_result()))
    out = reg.execute("query_table", {"sql": "SELECT ..."})
    assert not out.is_error
    lines = out.content.split("\n")
    assert lines[0] == (
        "fact_id | entity | field | value_num | doc_id | doc_source | doc_hash | conflict_group_id"
    )
    assert lines[1] == (
        "[#1] f1 | Harbor PM | late_fee | 75 | doc-1 | harbor/leases/unit-4b.md | h1 | cg1"
        " conflict=true"
    )
    assert lines[2].startswith("[#2] f2 ") and lines[2].endswith(" conflict=true")
    assert (
        lines[3]
        == "[#3] f3 | Harbor PM | grace_days | 5 | doc-1 | harbor/leases/unit-4b.md | h1 | "
    )
    assert "conflict=true" not in lines[3]
    assert len(lines) == 4


def test_query_table_sources_built_from_columns_like_fact_to_source():
    reg, _ = registry(FakeSession(result=fact_result()))
    out = reg.execute("query_table", {"sql": "SELECT ..."})
    expected = Fact(
        fact_id="f1",
        entity_id="e",
        entity="Harbor PM",
        field="late_fee",
        period_start=None,
        period_end=None,
        value=75,
        unit="USD",
        doc_id="doc-1",
        chunk_id=None,
        doc_source="harbor/leases/unit-4b.md",
        doc_hash="h1",
        conflict_group_id="cg1",
        extraction_job_id=None,
    ).to_source("harbor")
    assert out.sources[0] == expected
    assert [s.source_id for s in out.sources] == ["f1", "f2", "f3"]
    assert out.sources[1].uri == "harbor/addenda/late-fee.md"
    assert all(s.tenant_id == "harbor" for s in out.sources)


def test_query_table_sources_with_only_fact_id_column():
    res = QueryResult(
        columns=["fact_id", "value_num"], rows=[("f9", 60)], truncated=False, elapsed_ms=0
    )
    reg, _ = registry(FakeSession(result=res))
    out = reg.execute("query_table", {"sql": "SELECT fact_id, value_num FROM rag_facts"})
    assert out.content == "fact_id | value_num\n[#1] f9 | 60"
    assert out.sources == (
        Source(source_id="f9", uri="fact:f9", title="", span="fact:f9", tenant_id="harbor"),
    )


def test_query_table_source_falls_back_to_source_and_content_hash_columns():
    res = QueryResult(
        columns=["fact_id", "entity", "source", "content_hash"],
        rows=[("f1", "Harbor PM", "harbor/x.md", "zz")],
        truncated=False,
        elapsed_ms=0,
    )
    reg, _ = registry(FakeSession(result=res))
    out = reg.execute("query_table", {"sql": "SELECT ..."})
    s = out.sources[0]
    assert (s.uri, s.content_hash, s.title, s.doc_id) == ("harbor/x.md", "zz", "Harbor PM", None)


def test_query_table_rows_without_fact_id_get_no_marker_and_no_source():
    res = QueryResult(
        columns=["field", "n", "conflict_group_id"],
        rows=[("late_fee", 2, "cg1"), ("grace_days", 1, None)],
        truncated=False,
        elapsed_ms=0,
    )
    reg, _ = registry(FakeSession(result=res))
    out = reg.execute("query_table", {"sql": "SELECT field, count(*) ..."})
    assert (
        out.content
        == "field | n | conflict_group_id\nlate_fee | 2 | cg1 conflict=true\ngrace_days | 1 | "
    )
    assert "[#" not in out.content
    assert out.sources == ()


def test_query_table_null_fact_id_cell_gets_no_marker():
    res = QueryResult(
        columns=["fact_id", "n"], rows=[(None, 1), ("f2", 2)], truncated=False, elapsed_ms=0
    )
    reg, _ = registry(FakeSession(result=res))
    out = reg.execute("query_table", {"sql": "SELECT ... LEFT JOIN ..."})
    assert out.content == "fact_id | n\n | 1\n[#1] f2 | 2"
    assert [s.source_id for s in out.sources] == ["f2"]


def test_query_table_truncation_note():
    reg, _ = registry(FakeSession(result=fact_result(truncated=True)))
    out = reg.execute("query_table", {"sql": "SELECT * FROM rag_facts"})
    assert out.content.endswith("\n... (truncated at 3 rows; add a WHERE or LIMIT)")
    assert len(out.sources) == 3


def test_query_table_empty_result_is_just_the_header():
    res = QueryResult(columns=["fact_id", "value_num"], rows=[], truncated=False, elapsed_ms=0)
    reg, _ = registry(FakeSession(result=res))
    out = reg.execute("query_table", {"sql": "SELECT ... WHERE false"})
    assert out.content == "fact_id | value_num" and out.sources == ()


def test_query_table_rejected_sql_is_an_error_with_the_reason():
    reg, _ = registry(FakeSession(result=SqlRejected("not a single SELECT")))
    out = reg.execute("query_table", {"sql": "DELETE FROM rag_facts"})
    assert out.is_error
    assert out.content == "Query rejected: not a single SELECT"
    assert out.sources == ()


# ------------------------------------------------------------------ registry reuse rules


def test_shared_base_registry_is_copied_not_bound():
    base = ToolRegistry()

    @base.tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg = tenant_search_registry(FakeStore(), "harbor", registry=base)
    assert base.names() == ["add"] and base.tenant_id is None
    assert reg.names() == ["add", "search_documents", "fetch_document", "query_table"]
    with pytest.raises(ValueError):
        tenant_search_registry(FakeStore(), "summit", registry=reg)



def test_dictionary_line_omits_empty_unit_and_no_placeholder_leaks():
    from pathlib import Path

    from fde_kit.rag import tool
    from fde_kit.rag.tool import query_table_description

    entries = [DictEntry("late_fee", "Fee after grace", ""), DictEntry("grace_days", "Days", "d")]
    desc = query_table_description("base", entries)
    assert "- late_fee: Fee after grace" in desc and "late_fee ()" not in desc
    assert "- grace_days (d): Days" in desc
    # The model reads the tool's docstring; no template placeholder may be left in it.
    assert "<DICTIONARY>" not in Path(tool.__file__).read_text()
