import json
import os
from pathlib import Path

import pytest

from evals.make_replay_fixtures import FIXTURES, RUNS_DIR, generate
from evals.run_evals import (
    GOLDEN,
    GOLDEN_RAG,
    close_rag_store,
    load_cases,
    main,
    open_rag_store,
    run_case,
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
            assert any(chk["type"] == "regex" for chk in c["checks"]), c["id"]
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
