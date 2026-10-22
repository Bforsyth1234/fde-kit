"""Citations are produced by the tool layer and checked by the loop, never asserted by the model.

Offline: a mock provider and tools that return `ToolResult(content, sources=...)` directly.
"""

from fde_kit import Agent, Settings, Source, ToolRegistry, ToolResult
from fde_kit.citations import CitationLedger, cited, cited_numbers, unresolved_citations
from fde_kit.llm import MockProvider
from fde_kit.rag.store import Hit
from fde_kit.rag.tool import format_passages


def src(i: int) -> Source:
    return Source(source_id=f"chunk-{i}", uri=f"docs/{i}.md", title=f"Doc {i}", content_hash="ab")


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def search(query: str) -> ToolResult:
        """Search."""
        return ToolResult(
            "[#1] Doc 1\nLate fee is $50.\n\n[#2] Doc 2\nRent is due on the 1st.",
            sources=(src(1), src(2)),
        )

    @reg.tool
    def search_more(query: str) -> ToolResult:
        """Search again; returns one already-seen source and one new one."""
        return ToolResult("[#1] again\n\n[#2] Doc 3\nPets allowed.", sources=(src(2), src(3)))

    @reg.tool
    def plain(x: int) -> str:
        """A tool that returns a string, as every tool did before Source existed."""
        return str(x * 2)

    return reg


def run(script, reg=None):
    settings = Settings(provider="mock", otel_enabled=False)
    return Agent(reg or make_registry(), provider=MockProvider(script), settings=settings).run("q")


# ------------------------------------------------------------------ pure functions


def test_cited_numbers_parses_single_lists_and_adjacent_brackets():
    assert cited_numbers("fee is $50 [2]. due [1, 3][2] and [7]") == [2, 1, 3, 7]
    assert cited_numbers("no citations here, [not one] either") == []


def test_ledger_numbers_run_wide_and_rewrites_local_markers():
    ledger = CitationLedger()
    c1, n1 = ledger.add("[#1] a [#2] b", (src(1), src(2)))
    assert c1 == "[1] a [2] b" and n1 == [1, 2]
    # Second call: first source already seen keeps its number, new one gets 3.
    c2, n2 = ledger.add("[#1] again [#2] new [#9] bogus", (src(2), src(3)))
    assert c2 == "[2] again [3] new [#9] bogus" and n2 == [2, 3]
    assert [s.source_id for s in ledger.sources] == ["chunk-1", "chunk-2", "chunk-3"]


def test_cited_resolves_numbers_and_drops_unknown():
    sources = [src(1), src(2)]
    assert cited("see [2] and [2] and [5]", sources) == [src(2)]
    assert unresolved_citations("see [2] and [5]", sources) == [5]


# ------------------------------------------------------------------ through the loop


def test_tool_sources_reach_result_and_model_sees_run_wide_numbers():
    provider = MockProvider(
        [
            {"tool_calls": [{"name": "search", "input": {"query": "fee"}}]},
            {"tool_calls": [{"name": "search_more", "input": {"query": "pets"}}]},
            "The late fee is $50 [1] and pets are allowed [3].",
        ]
    )
    settings = Settings(provider="mock", otel_enabled=False)
    result = Agent(make_registry(), provider=provider, settings=settings).run("q")

    assert [s.source_id for s in result.sources] == ["chunk-1", "chunk-2", "chunk-3"]
    assert [s.source_id for s in result.cited] == ["chunk-1", "chunk-3"]
    assert result.fabricated_citations == []
    # The model was shown run-wide numbers, not the tool's local markers.
    second_call_results = provider.calls[2]["messages"][-1]["content"][0]["content"]
    assert "[2] again" in second_call_results and "[3] Doc 3" in second_call_results
    assert "[#" not in second_call_results
    # Each step records which run-wide numbers its tool call produced.
    assert result.steps[0].tool_results[0]["citations"] == [1, 2]
    assert result.steps[1].tool_results[0]["citations"] == [2, 3]
    assert result.steps[0].tool_results[0]["sources"][0]["source_id"] == "chunk-1"


def test_fabricated_citation_is_detected_not_resolved():
    result = run(
        [
            {"tool_calls": [{"name": "search", "input": {"query": "fee"}}]},
            "The fee is $50 [1] and the deposit is one month [4].",
        ]
    )
    assert [s.source_id for s in result.cited] == ["chunk-1"]
    assert result.fabricated_citations == [4]


def test_string_returning_tools_are_unchanged():
    result = run(
        [{"tool_calls": [{"name": "plain", "input": {"x": 21}}]}, "42"],
    )
    assert result.steps[0].tool_results[0]["content"] == "42"
    assert result.steps[0].tool_results[0]["sources"] == []
    assert result.sources == [] and result.cited == []


def test_source_dict_round_trip_drops_empty_fields():
    s = Source(source_id="x", uri="u", title="", doc_id=None, content_hash="h")
    d = s.to_dict()
    assert d == {"source_id": "x", "uri": "u", "content_hash": "h"}
    assert Source.from_dict(d) == s


# ------------------------------------------------------------------ rag formatting


def test_format_passages_numbers_hits_and_carries_hashes():
    hits = [
        Hit("c1", "d1", "summit/leases/101.md", "Lease 101", "Late fee $50", 0.9, 1, None, "hash1"),
        Hit("c2", "d1", "summit/leases/101.md", "Lease 101", "Rent due 1st", 0.8, 2, None, "hash2"),
    ]
    res = format_passages(hits, tenant_id="summit")
    assert res.content.startswith("[#1] Lease 101 (summit/leases/101.md)\nLate fee $50")
    assert "[#2] Lease 101" in res.content
    assert [s.source_id for s in res.sources] == ["c1", "c2"]
    assert res.sources[0].content_hash == "hash1" and res.sources[0].tenant_id == "summit"
    assert res.sources[0].doc_id == "d1" and res.sources[0].uri == "summit/leases/101.md"


# ------------------------------------------------------------------ eval checks


def test_cites_checks_resolve_through_numbers_not_prose():
    from evals.scorers import JudgeContext, cites, cites_any, cites_only

    ctx = JudgeContext(provider=None, prompt="q")
    good = run(
        [{"tool_calls": [{"name": "search", "input": {"query": "fee"}}]}, "It is $50 [1]."]
    )
    assert cites(good, "chunk-1", ctx).passed
    assert cites(good, "docs/1.md", ctx).passed  # by uri
    assert not cites(good, "chunk-2", ctx).passed  # returned, not cited
    assert cites_only(good, None, ctx).passed and cites_any(good, None, ctx).passed

    # Names the document in prose, cites nothing by number: not a citation.
    prose = run(
        [{"tool_calls": [{"name": "search", "input": {"query": "fee"}}]}, "Per docs/1.md, $50."]
    )
    assert not cites(prose, "docs/1.md", ctx).passed
    assert not cites_any(prose, None, ctx).passed

    fab = run(
        [{"tool_calls": [{"name": "search", "input": {"query": "fee"}}]}, "It is $50 [1][7]."]
    )
    assert cites(fab, "chunk-1", ctx).passed
    assert not cites_only(fab, None, ctx).passed
