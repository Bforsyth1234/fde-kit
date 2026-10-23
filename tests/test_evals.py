import json
import os
from pathlib import Path

import pytest

from evals.make_replay_fixtures import FIXTURES, RUNS_DIR, generate
from evals.run_evals import (
    EDGAR_TENANTS,
    GOLDEN,
    GOLDEN_EDGAR,
    GOLDEN_RAG,
    case_registry,
    close_rag_store,
    load_cases,
    main,
    open_rag_store,
    run_case,
    seed_edgar_offline,
    summarise,
)
from evals.scorers import SCORERS

PG_URL = os.environ.get("FDE_TEST_PG_URL")


def test_golden_set_has_twenty_five_valid_cases():
    cases = load_cases(GOLDEN)
    assert len(cases) == 25
    assert len({c["id"] for c in cases}) == 25, "duplicate ids"
    for c in cases:
        assert "mock" in c or "replay" in c, (
            f"{c['id']} has no mock script or replay file; offline eval would be vacuous"
        )


def test_every_case_passes_in_mock_mode(monkeypatch):
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    results = [run_case(c, "mock", judge=None) for c in load_cases(GOLDEN)]
    failing = [(r.id, r.details) for r in results if not r.passed]
    assert not failing, failing
    summary = summarise(results)
    assert summary["pass_rate"] == 1.0
    assert summary["by_category"]["calculator"]["passed"] == 8


def test_gate_exit_codes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    out = tmp_path / "r.json"
    assert main(["--provider", "mock", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["summary"]["total"] == 25

    # Flip one case's mock to a wrong answer -> below a 100% threshold -> exit 1.
    cases = load_cases(GOLDEN)
    cases[0]["mock"] = ["wrong"]
    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join(json.dumps(c) for c in cases))
    assert main(["--provider", "mock", "--golden", str(broken), "--threshold", "1.0"]) == 1
    assert main(["--provider", "mock", "--golden", str(broken), "--threshold", "0.9"]) == 0


def test_case_without_mock_or_replay_rejected(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "prompt": "p", "checks": []}))
    with pytest.raises(ValueError, match="needs a 'mock' script or a 'replay'"):
        load_cases(bad)
    bad.write_text(
        json.dumps({"id": "x", "prompt": "p", "checks": [], "replay": "evals/runs/nope.jsonl"})
    )
    with pytest.raises(ValueError, match="replay file not found"):
        load_cases(bad)


def test_replay_fixtures_regenerate_byte_identical(tmp_path: Path):
    """The committed run files are exactly what the generator produces: a format drift in
    fde_kit.record, or a hand edit of a fixture, fails here."""
    fresh = generate(tmp_path)
    assert [p.name for p in fresh] == [f"{fx.name}.jsonl" for fx in FIXTURES]
    for path in fresh:
        committed = RUNS_DIR / path.name
        assert committed.is_file(), f"{committed} missing: run python -m evals.make_replay_fixtures"
        assert path.read_bytes() == committed.read_bytes(), (
            f"{committed.name} differs from a fresh generation; "
            "run python -m evals.make_replay_fixtures"
        )
    # And the generator is deterministic on its own: a second run matches the first.
    again = generate(tmp_path / "again")
    for a, b in zip(fresh, again, strict=True):
        assert a.read_bytes() == b.read_bytes()


def test_replay_cases_pass_offline(monkeypatch):
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    cases = load_cases(GOLDEN, ids={"replay-01", "replay-02", "replay-03"})
    assert len(cases) == 3 and all("mock" not in c for c in cases)
    results = {r.id: r for r in (run_case(c, "mock", judge=None) for c in cases)}
    failing = [(r.id, r.details) for r in results.values() if not r.passed]
    assert not failing, failing
    assert results["replay-01"].replay == "evals/runs/replay-01.jsonl"
    assert results["replay-01"].cited == 1
    assert results["replay-02"].cited == 2
    assert results["replay-02"].tools_called == ["lookup_policy", "lookup_policy"]
    # replay-03 diverges on the first tool call: no step completes, no tool is recorded.
    assert results["replay-03"].turns == 0 and results["replay-03"].tools_called == []
    assert any("replay_diverged" in d for d in results["replay-03"].details)


def test_unknown_check_type_rejected(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "prompt": "p", "checks": [{"type": "nope"}]}))
    with pytest.raises(ValueError, match="unknown check type"):
        load_cases(bad)


def test_unknown_tenant_rejected(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "prompt": "p", "checks": [], "mock": ["a"], "tenant": "nobody"})
    )
    with pytest.raises(ValueError, match="unknown tenant 'nobody'"):
        load_cases(bad)


# --- the rag set (PR 3) ------------------------------------------------------------------


RAG_CATEGORIES = {"doc-lookup": 4, "table-fact": 4, "table-conflict": 2, "cross-tenant": 2}
RAG_TOOLS = ("search_documents", "fetch_document", "query_table")


def test_golden_rag_cases_are_well_formed():
    """Pure: shape only. Whether the scripts' SQL and queries return what the answers cite is
    the Postgres-gated test below."""
    from fde_kit.rag.demo_corpus import TENANTS, addendum_source

    cases = load_cases(GOLDEN_RAG)
    assert len(cases) == 12
    assert len({c["id"] for c in cases}) == 12, "duplicate ids"
    by_cat: dict[str, int] = {}
    for c in cases:
        assert c["tenant"] in TENANTS, c["id"]
        assert "mock" in c and "replay" not in c, f"{c['id']}: rag cases are scripted"
        assert {chk["type"] for chk in c["checks"]} <= set(SCORERS)
        assert any(chk["type"] == "cites_only" for chk in c["checks"]), c["id"]
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        # Every tool the script calls is one of the three tenant tools, with no tenant argument.
        for entry in c["mock"]:
            for call in entry.get("tool_calls", []) if isinstance(entry, dict) else []:
                assert call["name"] in RAG_TOOLS, c["id"]
                assert "tenant_id" not in call["input"], c["id"]
        if c["category"] == "table-conflict":
            conflict = [chk for chk in c["checks"] if chk["type"] == "conflict_reported"]
            assert len(conflict) == 1 and len(conflict[0]["value"]) == 2, c["id"]
            cited = {chk["value"] for chk in c["checks"] if chk["type"] == "cites"}
            assert addendum_source(c["tenant"]) in cited, c["id"]
        if c["category"] == "cross-tenant":
            # The other tenant's number must not appear in any format; wording goes to the judge.
            assert any(chk["type"] == "no_amount" for chk in c["checks"]), c["id"]
            assert any(chk["type"] == "judge" for chk in c["checks"]), c["id"]
    assert by_cat == RAG_CATEGORIES


@pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")
def test_golden_rag_passes_offline(monkeypatch):
    """Scripted model, real tools: every case's search and SQL, run under RLS against the
    seeded demo corpus, must return what the scripted answer cites."""
    import psycopg

    from fde_kit.rag import HashEmbedder, TenantStore, app_url_from_admin, create_tenant, migrate
    from fde_kit.rag.demo_corpus import TENANTS, seed

    with psycopg.connect(PG_URL) as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as exc:
            pytest.skip(f"pgvector unavailable: {exc}")
    migrate(PG_URL)
    for tid, t in TENANTS.items():
        create_tenant(PG_URL, tid, t["name"])
    with TenantStore(app_url_from_admin(PG_URL), HashEmbedder()) as store:
        seed(store)

    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    monkeypatch.setenv("FDE_RAG_ADMIN_URL", PG_URL)
    monkeypatch.delenv("FDE_RAG_URL", raising=False)
    monkeypatch.setenv("FDE_RAG_EMBEDDER", "hash")
    close_rag_store()
    open_rag_store()
    try:
        results = [run_case(c, "mock", judge=None) for c in load_cases(GOLDEN_RAG)]
    finally:
        close_rag_store()
    failing = [(r.id, r.details) for r in results if not r.passed]
    assert not failing, failing
    summary = summarise(results)
    assert summary["pass_rate"] == 1.0 and summary["fabricated_citations"] == 0
    assert {c: v["passed"] for c, v in summary["by_category"].items()} == RAG_CATEGORIES
    by_id = {r.id: r for r in results}
    assert all(by_id[i].cited == 2 for i in ("rag-conflict-01", "rag-conflict-02"))
    assert all(by_id[i].cited == 0 for i in ("rag-xtenant-01", "rag-xtenant-02"))


# --- the edgar set (PR 4) ----------------------------------------------------------------


EDGAR_CATEGORIES = {
    "table-fact": 3,
    "table-conflict": 3,
    "precision": 2,
    "doc-lookup": 2,
    "cross-tenant": 2,
}
EDGAR_DOC_PREFIX = "https://www.sec.gov/Archives/edgar/data/"


def _manifest_documents() -> set[str]:
    from fde_kit.rag.adapters import load_manifest

    return {url for e in load_manifest() for url in e.documents}


def test_golden_edgar_cases_are_well_formed():
    """Pure: shape only, plus every cited URL is a pinned manifest document. Whether the
    scripts return what the answers cite is the Postgres-gated test below."""
    cases = load_cases(GOLDEN_EDGAR)
    assert len(cases) == 12
    assert len({c["id"] for c in cases}) == 12, "duplicate ids"
    docs = _manifest_documents()
    by_cat: dict[str, int] = {}
    for c in cases:
        assert c["tenant"] in EDGAR_TENANTS, c["id"]
        assert "mock" in c and "replay" not in c, f"{c['id']}: edgar cases are scripted"
        assert any(chk["type"] == "cites_only" for chk in c["checks"]), c["id"]
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        for entry in c["mock"]:
            for call in entry.get("tool_calls", []) if isinstance(entry, dict) else []:
                assert call["name"] in RAG_TOOLS, c["id"]
                assert "tenant_id" not in call["input"], c["id"]
                if call["name"] == "query_table":
                    # Citable rows: fact_id plus the filing URL as doc_source, stable order.
                    sql = call["input"]["sql"]
                    assert "f.fact_id" in sql and "d.source AS doc_source" in sql, c["id"]
                    assert "ORDER BY" in sql, c["id"]
        cited = [chk["value"] for chk in c["checks"] if chk["type"] == "cites"]
        for url in cited:
            assert url.startswith(EDGAR_DOC_PREFIX) and url in docs, (c["id"], url)
        if c["category"] == "table-conflict":
            conflict = [chk for chk in c["checks"] if chk["type"] == "conflict_reported"]
            assert len(conflict) == 1 and len(conflict[0]["value"]) >= 2, c["id"]
            assert len(cited) >= 2, f"{c['id']}: a conflict cites every filing"
        if c["category"] == "precision":
            assert any(chk["type"] == "amount" for chk in c["checks"]), c["id"]
            assert any(chk["type"] == "judge" for chk in c["checks"]), c["id"]
        if c["category"] == "cross-tenant":
            assert not cited, c["id"]
            assert any(chk["type"] == "no_amount" for chk in c["checks"]), c["id"]
            assert any(chk["type"] == "judge" for chk in c["checks"]), c["id"]
    assert by_cat == EDGAR_CATEGORIES
    assert {c["tenant"] for c in cases} == set(EDGAR_TENANTS)


@pytest.fixture(scope="module")
def edgar_store():
    """Seed pharma-a / pharma-b once from the offline fixtures (idempotent) and open the run's
    store against them with the hash embedder, as CI does."""
    if not PG_URL:
        pytest.skip("FDE_TEST_PG_URL not set")
    import psycopg

    with psycopg.connect(PG_URL) as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as exc:
            pytest.skip(f"pgvector unavailable: {exc}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FDE_OTEL_ENABLED", "false")
        mp.setenv("FDE_RAG_ADMIN_URL", PG_URL)
        mp.delenv("FDE_RAG_URL", raising=False)
        mp.setenv("FDE_RAG_EMBEDDER", "hash")
        close_rag_store()
        try:
            # The helper migrates before it opens the run's store (the app role must exist).
            seed_edgar_offline(PG_URL)
            yield open_rag_store()
        finally:
            close_rag_store()


def test_golden_edgar_passes_offline(edgar_store):
    """Scripted model, real tools: every case's search and SQL, run under RLS against the
    offline-seeded EDGAR corpus, must return what the scripted answer cites."""
    results = [run_case(c, "mock", judge=None) for c in load_cases(GOLDEN_EDGAR)]
    failing = [(r.id, r.details) for r in results if not r.passed]
    assert not failing, failing
    summary = summarise(results)
    assert summary["total"] == 12 and summary["passed"] == 12
    assert summary["pass_rate"] == 1.0 and summary["fabricated_citations"] == 0
    assert {c: v["passed"] for c, v in summary["by_category"].items()} == EDGAR_CATEGORIES
    by_id = {r.id: r for r in results}
    assert all(by_id[i].cited == 0 for i in ("edgar-xtenant-01", "edgar-xtenant-02"))


def _tool_outputs(case: dict) -> list[str]:
    """Run the case's scripted tool calls for real (no model) and return each tool's content."""
    reg = case_registry(case)
    out = []
    for entry in case["mock"]:
        for call in entry.get("tool_calls", []) if isinstance(entry, dict) else []:
            res = reg.execute(call["name"], call["input"])
            assert not res.is_error, (case["id"], res.content)
            out.append(res.content)
    return out


def test_edgar_precision_cases_have_no_conflict_in_tool_output(edgar_store):
    """The scorers only see the answer, so check the tool side directly: a rounded repeat of
    the same figure (Vertex FY2019 revenue, Moderna FY2020 net loss) is not a conflict, while
    the real restatements in the table-conflict cases are."""
    cases = load_cases(GOLDEN_EDGAR)
    precision = [c for c in cases if c["category"] == "precision"]
    conflicts = [c for c in cases if c["category"] == "table-conflict"]
    assert len(precision) == 2 and len(conflicts) == 3
    for c in precision:
        content = "\n".join(_tool_outputs(c))
        rows = [line for line in content.splitlines() if line.startswith("[#")]
        assert len(rows) >= 2, f"{c['id']}: expected repeats across filings: {content}"
        # Column 3 is value_num: the filings round the same figure differently.
        assert len({line.split(" | ")[3] for line in rows}) >= 2, (c["id"], content)
        assert "conflict=true" not in content, f"{c['id']}: {content}"
    for c in conflicts:
        content = "\n".join(_tool_outputs(c))
        rows = [line for line in content.splitlines() if line.startswith("[#")]
        assert rows and all(line.endswith("conflict=true") for line in rows), (c["id"], content)
