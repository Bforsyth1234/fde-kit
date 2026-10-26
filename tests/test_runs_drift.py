"""`fde runs drift` / `fde_kit.rag.drift`: a real recorded run checked against a live database.

Same gate and fixtures style as tests/test_rag_facts.py: FDE_TEST_PG_URL must point at an admin
URL on a server with pgvector >= 0.8, or everything here skips. Each test re-seeds the Harbor
demo tenant, records a real run with MockProvider (search_documents, fetch_document,
query_table and the non-database lookup_policy), changes the database, and checks the report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
import pytest

from fde_kit import Agent, Settings, cli
from fde_kit.builtin_tools import registry as builtin_registry
from fde_kit.llm import MockProvider
from fde_kit.rag import (
    FactRow,
    HashEmbedder,
    TenantStore,
    app_url_from_admin,
    create_tenant,
    migrate,
    tenant_search_registry,
)
from fde_kit.rag.demo_corpus import TENANTS, seed
from fde_kit.rag.drift import TenantError, check_drift, recorded_fact
from fde_kit.rag.schema import drop_all
from fde_kit.record import RunRecord

PG_URL = os.environ.get("FDE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")

TID = "harbor"
NAME = TENANTS[TID]["name"]
FIRST_LEASE = f"{TID}/leases/" + TENANTS[TID]["units"][0][0].lower().replace(" ", "-") + ".md"
FACT_SQL = (
    "SELECT f.fact_id, f.field, f.value_num, f.value_text, f.unit FROM rag_facts f "
    "WHERE f.field IN ('grace_days', 'late_fee') ORDER BY f.field, f.value_num"
)


@pytest.fixture(scope="module")
def admin_url() -> str:
    assert PG_URL
    with psycopg.connect(PG_URL) as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as exc:
            pytest.skip(f"pgvector unavailable: {exc}")
        ver = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if ver and tuple(int(x) for x in ver[0].split(".")[:2]) < (0, 8):
        pytest.skip(f"pgvector {ver[0]} < 0.8 (no iterative index scans)")
    drop_all(PG_URL)
    migrate(PG_URL)
    for tid, t in TENANTS.items():
        create_tenant(PG_URL, tid, t["name"])
    return PG_URL


@pytest.fixture(scope="module")
def store(admin_url: str):
    with TenantStore(app_url_from_admin(admin_url), HashEmbedder()) as s:
        yield s


def _doc_id(store: TenantStore, source: str) -> str:
    with store.session(TID) as s:
        row = s.conn.execute(
            "SELECT doc_id::text FROM rag_documents WHERE source = %s AND superseded_by IS NULL",
            (source,),
        ).fetchone()
    assert row
    return row[0]


@pytest.fixture
def run(store: TenantStore, tmp_path: Path) -> RunRecord:
    """A fresh Harbor tenant and a real recorded run over it."""
    seed(store, [TID])
    lease = _doc_id(store, FIRST_LEASE)
    script = [
        {
            "tool_calls": [
                {"name": "search_documents", "input": {"query": "Dana Whitfield late fee", "k": 3}}
            ]
        },
        {
            "tool_calls": [
                {
                    "name": "fetch_document",
                    "input": {"doc_id": lease, "char_start": 0, "char_end": 400},
                }
            ]
        },
        {"tool_calls": [{"name": "query_table", "input": {"sql": FACT_SQL}}]},
        {"tool_calls": [{"name": "lookup_policy", "input": {"topic": "sla"}}]},
        "The late fee is $75 [1] after a grace period [5].",
    ]
    reg = tenant_search_registry(store, TID, registry=builtin_registry, cached=False)
    settings = Settings(provider="mock", otel_enabled=False)
    runs = tmp_path / "runs"
    agent = Agent(reg, provider=MockProvider(script), settings=settings)
    result = agent.run("What is the late fee?", record=runs / "placeholder.jsonl")
    assert result.stopped_reason == "end_turn"
    rec = RunRecord.load(result.record_path)
    # Name the file after the run so `fde runs drift RUN_ID --dir` finds it either way.
    Path(result.record_path).rename(runs / f"{rec.run_id}.jsonl")
    assert rec.header["tenant_id"] == TID
    return rec


def _by_tool(report, tool: str):
    return [i for i in report.items if i.tool == tool]


def _fact(report, field: str, value: str):
    for i in _by_tool(report, "query_table"):
        if field in i.title and i.then_value == value:
            return i
    raise AssertionError(f"no {field}={value} in {[(i.title, i.then_value) for i in report.items]}")


def _grace_fact_id(rec: RunRecord) -> str:
    row = next(r for r in rec.tool_rows() if r["name"] == "query_table")
    for src in row["sources"]:
        if "grace_days" in src["title"]:
            return src["source_id"]
    raise AssertionError("grace_days not in the recorded query")


# --------------------------------------------------------------------------- library


def test_no_changes_everything_unchanged(store, run):
    report = check_drift(run, store)
    assert report.tenant_id == TID
    assert not report.drifted
    statuses = {i.tool: set() for i in report.items}
    for i in report.items:
        statuses[i.tool].add(i.status)
    assert statuses == {
        "search_documents": {"unchanged"},
        "fetch_document": {"unchanged"},
        "query_table": {"unchanged"},
        "lookup_policy": {"unchecked"},
    }
    # Numbers are the run's citation numbers: first-seen order, deduplicated by source_id.
    assert [i.number for i in report.items] == list(range(1, len(report.items) + 1))
    assert len(_by_tool(report, "search_documents")) == 3
    # The recorded value and conflict state were read back from the tool row.
    grace = _fact(report, "grace_days", str(TENANTS[TID]["grace_days"]))
    assert grace.then_conflict is False and grace.now_conflict is False
    late = _by_tool(report, "query_table")
    assert all(i.then_conflict for i in late if "late_fee" in i.title)  # seeded conflict


def test_reingested_document_is_superseded_with_both_hashes(store, run):
    lease = _doc_id(store, FIRST_LEASE)
    with store.session(TID) as s:
        old_body = s.fetch(lease, 0, 10**6).text
        s.ingest(FIRST_LEASE, old_body + "\n\nAmended: the late fee is now $95.", title="Lease")
    new = _doc_id(store, FIRST_LEASE)
    assert new != lease

    report = check_drift(run, store)
    assert report.drifted
    (sl,) = _by_tool(report, "fetch_document")
    assert sl.status == "superseded"
    assert sl.then_hash and sl.now_hash and sl.then_hash != sl.now_hash
    assert (sl.then_version, sl.now_version, sl.now_doc_id) == (1, 2, new)
    assert "v1" in sl.detail and "v2" in sl.detail

    # Passages from that document: their chunks were deleted by the re-ingest, judged by doc.
    chunks = [i for i in _by_tool(report, "search_documents") if i.doc_id == lease]
    others = [i for i in _by_tool(report, "search_documents") if i.doc_id != lease]
    assert chunks, "the search should have returned a passage of the re-ingested lease"
    for i in chunks:
        assert i.status == "superseded" and i.now_doc_id == new
        assert i.then_hash != i.now_hash
    assert all(i.status == "unchanged" for i in others)

    # Facts extracted from the old version still exist but point at v1.
    grace = _fact(report, "grace_days", str(TENANTS[TID]["grace_days"]))
    assert grace.status == "superseded" and "still points at v1" in grace.detail


def test_deleted_fact_and_document_are_gone(store, run):
    lease = _doc_id(store, FIRST_LEASE)
    report = check_drift(run, store)
    assert not report.drifted
    # The addendum's revised late_fee: a fact from a document other than the first lease.
    addendum = next(i for i in _by_tool(report, "query_table") if i.doc_id != lease)
    with store.session(TID) as s:
        s.conn.execute("DELETE FROM rag_facts WHERE fact_id = %s::uuid", (_grace_fact_id(run),))
        s.conn.execute("DELETE FROM rag_documents WHERE doc_id = %s::uuid", (addendum.doc_id,))

    report = check_drift(run, store)
    assert report.drifted
    grace = _fact(report, "grace_days", str(TENANTS[TID]["grace_days"]))
    assert grace.status == "gone" and "no longer exists" in grace.detail
    revised = next(i for i in report.items if i.source_id == addendum.source_id)
    assert revised.status == "gone" and "document" in revised.detail
    passages = [i for i in _by_tool(report, "search_documents") if i.doc_id == addendum.doc_id]
    assert passages, "the search should have returned a passage of the addendum"
    assert all(i.status == "gone" for i in passages)


def test_changed_value_and_new_conflict(store, run):
    grace_id = _grace_fact_id(run)
    days = int(TENANTS[TID]["grace_days"])
    with store.session(TID) as s:
        s.conn.execute(
            "UPDATE rag_facts SET value_num = %s WHERE fact_id = %s::uuid", (days + 4, grace_id)
        )
    report = check_drift(run, store)
    grace = next(i for i in report.items if i.source_id == grace_id)
    assert grace.status == "value_changed"
    assert (grace.then_value, grace.now_value) == (str(days), str(days + 4))
    assert f"value {days} -> {days + 4}" in grace.detail
    assert report.drifted

    # Put the value back, then have another source disagree: same value, new conflict.
    with store.session(TID) as s:
        s.conn.execute(
            "UPDATE rag_facts SET value_num = %s WHERE fact_id = %s::uuid", (days, grace_id)
        )
        s.add_facts([FactRow(NAME, "grace_days", days + 10, "days")])
    report = check_drift(run, store)
    grace = next(i for i in report.items if i.source_id == grace_id)
    assert grace.status == "new_conflict"
    assert (grace.then_conflict, grace.now_conflict) == (False, True)
    assert report.drifted
    # The late_fee facts were already conflicting when the run saw them: not new.
    assert all(
        i.status == "unchanged" for i in _by_tool(report, "query_table") if "late_fee" in i.title
    )


def test_other_tenant_is_refused(store, run):
    with pytest.raises(TenantError, match="recorded for tenant 'harbor'"):
        check_drift(run, store, "summit")
    # Same tenant, explicitly: fine.
    assert check_drift(run, store, TID).tenant_id == TID
    # No tenant anywhere: refused with a pointer to --tenant.
    headless = RunRecord(header={**run.header, "tenant_id": None}, rows=run.rows)
    with pytest.raises(TenantError, match="--tenant"):
        check_drift(headless, store)


def test_non_rag_source_is_unchecked_not_an_error(store, run):
    report = check_drift(run, store)
    (pol,) = _by_tool(report, "lookup_policy")
    assert pol.status == "unchecked" and "not a database tool" in pol.detail
    assert not pol.drifted


def test_recorded_fact_reads_value_and_conflict_back():
    row = {
        "content": "fact_id | field | value_num | value_text | unit\n"
        "[#1] f-1 | late_fee | 75 |  | USD (source: Lease (a | b)) conflict=true\n"
        "[#2] f-2 | grace_days | 5 |  | days (source: Lease)\n"
        "f-3 | grace_days | 9 |  | days (unverified: fact_id not found)"
    }
    one = recorded_fact(row, "f-1")
    assert one and one.conflict and one.columns["value_num"] == "75"
    two = recorded_fact(row, "f-2")
    assert two and not two.conflict and two.columns["unit"] == "days"
    assert recorded_fact(row, "f-3") is None
    assert recorded_fact({"content": "count\n3"}, "f-1") is None


# --------------------------------------------------------------------------- CLI


def _cli(capsys, monkeypatch, admin_url, *argv):
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setenv("FDE_RAG_ADMIN_URL", admin_url)
    monkeypatch.delenv("FDE_RAG_URL", raising=False)
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_table_json_and_exit_codes(store, run, admin_url, tmp_path, capsys, monkeypatch):
    d = str(tmp_path / "runs")
    code, out = _cli(capsys, monkeypatch, admin_url, "runs", "drift", run.run_id, "--dir", d)
    assert code == 0, out
    assert f"drift: {run.run_id} (tenant harbor)" in out
    assert "unchanged" in out and "unchecked" in out and "not a database tool" in out
    assert "-> no drift" in out

    code, out = _cli(
        capsys, monkeypatch, admin_url, "runs", "drift", run.run_id, "--dir", d, "--json"
    )
    assert code == 0
    data = json.loads(out)
    assert data["run_id"] == run.run_id and data["drifted"] is False
    assert {s["status"] for s in data["sources"]} == {"unchanged", "unchecked"}

    with store.session(TID) as s:
        s.conn.execute("DELETE FROM rag_facts WHERE fact_id = %s::uuid", (_grace_fact_id(run),))
    code, out = _cli(capsys, monkeypatch, admin_url, "runs", "drift", run.run_id, "--dir", d)
    assert code == 1
    assert "gone" in out and "-> drift" in out
    code, out = _cli(
        capsys, monkeypatch, admin_url, "runs", "drift", run.run_id, "--dir", d, "--json"
    )
    assert code == 1
    assert json.loads(out)["counts"]["gone"] == 1

    # Usage errors: another tenant, unknown run.
    code, out = _cli(
        capsys, monkeypatch, admin_url, "runs", "drift", run.run_id, "--dir", d,
        "--tenant", "summit",
    )  # fmt: skip
    assert code == 2 and "refused" in out and "row-level security" in out
    code, out = _cli(capsys, monkeypatch, admin_url, "runs", "drift", "r-nope", "--dir", d)
    assert code == 2 and "unknown run" in out
