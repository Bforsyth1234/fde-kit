"""The EDGAR corpus in Postgres: `fde rag seed-edgar --offline` end to end.

Seeds `pharma-a` (Pfizer, Gilead, Vertex) and `pharma-b` (Moderna, Regeneron) from the committed
offline fixtures through the same helper the CLI uses, then checks counts, idempotence, the
conflict groups the real restatements produce (and the precision-only repeats do not), fact
provenance down to the EX-13, and tenant isolation. No network: the client is offline.

Needs Postgres with pgvector >= 0.8 at FDE_TEST_PG_URL (an admin/superuser URL), like
tests/test_rag_isolation.py.
"""

from __future__ import annotations

import os
import re
from datetime import date

import psycopg
import pytest

from fde_kit.cli import (
    EDGAR_TENANTS,
    edgar_client,
    edgar_filings,
    ingest_edgar,
    main,
    seed_edgar,
)
from fde_kit.rag import (
    HashEmbedder,
    TenantStore,
    app_url_from_admin,
    check_posture,
    create_tenant,
    drop_tenant,
    migrate,
    tenant_search_registry,
)
from fde_kit.rag.adapters import filing_documents, load_manifest

PG_URL = os.environ.get("FDE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")

PFIZER = "PFIZER INC"
GILEAD = "GILEAD SCIENCES, INC."
VERTEX = "VERTEX PHARMACEUTICALS INC / MA"
MODERNA = "Moderna, Inc."
REGENERON = "REGENERON PHARMACEUTICALS, INC."

REV = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"

PFIZER_REVENUE_SQL = (
    "SELECT f.fact_id, e.canonical_name, f.value_num FROM rag_facts f "
    "JOIN rag_entities e ON e.entity_id = f.entity_id "
    "WHERE e.canonical_name ILIKE '%pfizer%' AND f.field = 'revenue'"
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
    migrate(PG_URL)
    for tid in EDGAR_TENANTS:  # a fresh start, so the counts below are the first ingest's
        drop_tenant(PG_URL, tid)
    return PG_URL


@pytest.fixture(scope="module")
def app_url(admin_url: str) -> str:
    return app_url_from_admin(admin_url)


@pytest.fixture(scope="module")
def store(app_url: str):
    with TenantStore(app_url, HashEmbedder()) as s:
        yield s


@pytest.fixture(scope="module")
def seeded(admin_url: str, store: TenantStore):
    """First ingest of both tenants, offline, through the CLI's helper."""
    return seed_edgar(store, admin_url, offline=True)


def _count(store: TenantStore, tenant: str, sql: str):
    with store.session(tenant) as s:
        return s.query(sql).rows


def _fy(store: TenantStore, tenant: str, entity: str, field: str, year: int):
    with store.session(tenant) as s:
        return [
            f for f in s.facts(entity=entity, field=field)
            if f.period_end is not None and f.period_end.year == year
        ]


def _groups(facts) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for f in facts:
        if f.conflict_group_id:
            out.setdefault(f.conflict_group_id, set()).add(int(f.value))
    return out


# ------------------------------------------------------------------ counts


def test_first_ingest_reports(seeded):
    a, b = seeded["pharma-a"], seeded["pharma-b"]
    assert (a.entities, a.documents, a.documents_changed, a.facts) == (3, 10, 10, 54)
    assert (b.entities, b.documents, b.documents_changed, b.facts) == (2, 6, 6, 36)
    assert a.job_id and b.job_id


def test_counts(seeded, store: TenantStore):
    for tenant, entities, docs, facts in (("pharma-a", 3, 10, 54), ("pharma-b", 2, 6, 36)):
        rows = _count(store, tenant, "SELECT count(*) FROM rag_entities WHERE kind = 'company'")
        assert rows[0][0] == entities
        rows = _count(store, tenant, "SELECT count(*) FROM rag_documents")
        assert rows[0][0] == docs
        assert _count(store, tenant, "SELECT count(*) FROM rag_facts")[0][0] == facts
    per_ticker = dict(
        _count(
            store,
            "pharma-a",
            "SELECT metadata->>'ticker', count(*) FROM rag_documents GROUP BY 1",
        )
    )
    assert per_ticker == {"PFE": 4, "GILD": 3, "VRTX": 3}
    per_ticker = dict(
        _count(
            store,
            "pharma-b",
            "SELECT metadata->>'ticker', count(*) FROM rag_documents GROUP BY 1",
        )
    )
    assert per_ticker == {"MRNA": 3, "REGN": 3}
    # 3 filings x 6 facts per company
    for tenant in EDGAR_TENANTS:
        rows = _count(
            store,
            tenant,
            "SELECT e.canonical_name, count(*) FROM rag_facts f "
            "JOIN rag_entities e ON e.entity_id = f.entity_id GROUP BY 1",
        )
        assert all(n == 18 for _, n in rows), rows


def test_rerun_writes_nothing(seeded, store: TenantStore):
    client = edgar_client(offline=True)
    for tenant in EDGAR_TENANTS:
        before = _count(store, tenant, "SELECT count(*) FROM rag_facts")[0][0]
        docs_before = _count(store, tenant, "SELECT count(*) FROM rag_documents")[0][0]
        r = ingest_edgar(store, tenant, client)
        assert r.documents_changed == 0 and r.facts == 0 and r.job_id == ""
        assert r.documents == docs_before
        assert r.conflict_groups == seeded[tenant].conflict_groups
        assert _count(store, tenant, "SELECT count(*) FROM rag_facts")[0][0] == before
        assert _count(store, tenant, "SELECT count(*) FROM rag_documents")[0][0] == docs_before


# ------------------------------------------------------------------ conflicts


@pytest.mark.parametrize(
    ("tenant", "entity", "field", "year", "values"),
    [
        ("pharma-a", PFIZER, "revenue", 2019, {51_750_000_000, 41_172_000_000, 40_905_000_000}),
        ("pharma-a", GILEAD, "rd_expense", 2019, {9_106_000_000, 4_055_000_000}),
        ("pharma-b", REGENERON, "revenue", 2019, {7_863_400_000, 6_557_600_000}),
    ],
)
def test_restatements_form_one_conflict_group(seeded, store, tenant, entity, field, year, values):
    facts = _fy(store, tenant, entity, field, year)
    assert len(facts) == 3  # one per filing, every filing kept as a source
    groups = _groups(facts)
    assert len(groups) == 1, groups
    assert next(iter(groups.values())) == values
    assert all(f.conflict_group_id for f in facts)


def test_gilead_rd_conflict_is_a_tag_switch(seeded, store):
    facts = _fy(store, "pharma-a", GILEAD, "rd_expense", 2019)
    tags = {int(f.value): f.source_ref.split("@")[0] for f in facts}
    assert tags[9_106_000_000] == "us-gaap:ResearchAndDevelopmentExpense"
    assert tags[4_055_000_000] == (
        "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
    )


@pytest.mark.parametrize(
    ("tenant", "entity", "field", "year"),
    [
        ("pharma-a", VERTEX, "revenue", 2019),
        ("pharma-b", MODERNA, "net_income", 2020),
    ],
)
def test_precision_only_differences_form_no_group(seeded, store, tenant, entity, field, year):
    facts = _fy(store, tenant, entity, field, year)
    assert len(facts) >= 2
    assert len({int(f.value) for f in facts}) == 2  # the values differ, but only in rounding
    assert all(f.conflict_group_id is None for f in facts)


# ------------------------------------------------------------------ provenance


def test_pfizer_facts_cite_the_document_with_the_statements(seeded, store):
    client = edgar_client(offline=True)
    by_fy = {f.fiscal_year: f for f in edgar_filings("pharma-a", ["PFE"])}
    assert sorted(by_fy) == [2019, 2020, 2021]
    docs = {fy: filing_documents(client, f) for fy, f in by_fy.items()}
    ex13 = [u for u, t in docs[2019] if t.upper().startswith("EX-13")]
    assert len(ex13) == 1
    assert not any(t.upper().startswith("EX-13") for fy in (2020, 2021) for _, t in docs[fy])

    with store.session("pharma-a") as s:
        facts = s.facts(entity=PFIZER)
        assert len(facts) == 18
        for fy, filing in by_fy.items():
            own = [f for f in facts if f.source_ref.endswith(f"@{filing.accession}")]
            assert len(own) == 3 * (fy - 2018)  # 3 fields x the FYs 2019.. it reports
            expected = ex13[0] if fy == 2019 else docs[fy][0][0]
            for f in own:
                assert f.doc_source == expected
                assert s.fetch(f.doc_id, 0, 10).source == expected
    assert ex13[0].endswith("pfe-exhibit13x12312019.htm")


def test_every_fact_has_source_ref_and_precision(seeded, store):
    accessions: dict[str, set[str]] = {}
    for e in load_manifest():
        accessions.setdefault(e.tenant, set()).add(e.accession)
    pat = re.compile(r"^us-gaap:[A-Za-z]+@(?P<acc>[0-9-]+)$")
    for tenant in EDGAR_TENANTS:
        with store.session(tenant) as s:
            facts = s.facts()
        assert facts
        for f in facts:
            m = pat.match(f.source_ref)
            assert m, f.source_ref
            assert m["acc"] in accessions[tenant]
            assert f.precision is not None and f.precision >= 1
            assert f.unit == "USD"
            assert isinstance(f.period_end, date)
            assert f.doc_id


# ------------------------------------------------------------------ isolation


def test_pharma_b_sees_only_its_companies(seeded, store):
    rows = _count(store, "pharma-b", "SELECT DISTINCT e.canonical_name FROM rag_entities e")
    assert sorted(r[0] for r in rows) == sorted([MODERNA, REGENERON])


def test_query_table_as_pharma_b_finds_no_pfizer(seeded, store):
    out = tenant_search_registry(store, "pharma-b").execute(
        "query_table", {"sql": PFIZER_REVENUE_SQL}
    )
    assert not out.is_error, out.content
    assert "PFIZER" not in out.content.upper()
    assert "51750000000" not in out.content
    # Control: the same statement as pharma-a does see Pfizer.
    out = tenant_search_registry(store, "pharma-a").execute(
        "query_table", {"sql": PFIZER_REVENUE_SQL}
    )
    assert not out.is_error and "PFIZER" in out.content


def test_posture_clean(seeded, app_url):
    with psycopg.connect(app_url) as conn:
        assert check_posture(conn) == []


# ------------------------------------------------------------------ CLI


def test_cli_seed_edgar_offline(seeded, admin_url, store, capsys):
    rc = main(
        ["rag", "seed-edgar", "--offline", "--admin-url", admin_url, "--embedder", "hash"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pharma-a" in out and "pharma-b" in out
    # Idempotent through the CLI too.
    assert _count(store, "pharma-a", "SELECT count(*) FROM rag_facts")[0][0] == 54
    assert _count(store, "pharma-b", "SELECT count(*) FROM rag_facts")[0][0] == 36


def test_cli_ingest_edgar_online_without_contact(monkeypatch, capsys):
    monkeypatch.setenv("FDE_EDGAR_USER_AGENT", "")
    rc = main(["rag", "ingest", "--tenant", "pharma-a", "--adapter", "edgar"])
    assert rc == 2
    assert "FDE_EDGAR_USER_AGENT" in capsys.readouterr().out


def test_cli_edgar_drift_without_contact(monkeypatch, capsys):
    monkeypatch.setenv("FDE_EDGAR_USER_AGENT", "")
    assert main(["rag", "edgar-drift"]) == 2
    assert "FDE_EDGAR_USER_AGENT" in capsys.readouterr().out


def test_cli_ingest_edgar_offline_one_ticker(seeded, admin_url, store, capsys):
    rc = main(
        [
            "rag", "ingest", "--tenant", "pharma-b", "--adapter", "edgar", "--ticker", "MRNA",
            "--offline", "--url", app_url_from_admin(admin_url), "--embedder", "hash",
        ]
    )
    assert rc == 0
    assert "pharma-b" in capsys.readouterr().out
    assert _count(store, "pharma-b", "SELECT count(*) FROM rag_facts")[0][0] == 36


def test_cli_ingest_unknown_ticker(capsys):
    rc = main(["rag", "ingest", "--tenant", "pharma-b", "--adapter", "edgar", "--ticker",
               "PFE", "--offline"])
    assert rc == 2
    assert "PFE" in capsys.readouterr().out


def test_cli_ingest_local_files(admin_url, store, tmp_path, capsys):
    tid = "edgar-local-test"
    (tmp_path / "notes.md").write_text("# Notes\n\nThe review is due on Friday.\n")
    (tmp_path / "page.html").write_text("<html><body><p>Quarterly summary</p></body></html>")
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4\n")
    try:
        create_tenant(admin_url, tid, "Local files test")
        rc = main(
            [
                "rag", "ingest", "--tenant", tid, "--adapter", "local_files",
                "--dir", str(tmp_path), "--entity", "Test Co",
                "--url", app_url_from_admin(admin_url), "--embedder", "hash",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert tid in out and "skipped scan.pdf" in out
        rows = _count(store, tid, "SELECT source FROM rag_documents ORDER BY source")
        assert [r[0] for r in rows] == ["notes.md", "page.html"]
        rows = _count(store, tid, "SELECT canonical_name FROM rag_entities")
        assert [r[0] for r in rows] == ["Test Co"]
        assert main(["rag", "ingest", "--tenant", tid, "--adapter", "local_files"]) == 2
    finally:
        drop_tenant(admin_url, tid)


# --- provenance from the database, whatever the SQL selected (live run r-1feef1c45323)


def test_query_table_citations_carry_the_filing_even_when_sql_selects_no_document(seeded, store):
    """The live model selected fact_id and values only. Every citation must still resolve to
    its SEC filing (URL + hash), the model must see which filing each row came from, and the
    three Pfizer FY2019 revenue rows must be marked as conflicting without selecting
    conflict_group_id."""
    out = tenant_search_registry(store, "pharma-a").execute(
        "query_table",
        {
            "sql": "SELECT f.fact_id, f.value_num FROM rag_facts f "
            "JOIN rag_entities e ON e.entity_id = f.entity_id "
            "WHERE e.canonical_name = 'PFIZER INC' AND f.field = 'revenue' "
            "AND f.period_end = '2019-12-31' ORDER BY f.value_num DESC"
        },
    )
    assert not out.is_error
    rows = out.content.splitlines()[1:]
    assert [r.split(" | ")[1].split(" ")[0] for r in rows] == [
        "51750000000",
        "41172000000",
        "40905000000",
    ]
    assert all(r.endswith(" conflict=true") for r in rows)
    assert f"(source: PFIZER INC 10-K FY2019 (EX-13); reported as {REV})" in rows[0]
    assert f"(source: PFIZER INC 10-K FY2020 (10-K); reported as {REV})" in rows[1]
    assert f"(source: PFIZER INC 10-K FY2021 (10-K); reported as {REV})" in rows[2]
    assert len(out.sources) == 3
    for src in out.sources:
        assert src.uri.startswith("https://www.sec.gov/Archives/edgar/data/78003/")
        assert src.content_hash and src.doc_id and src.tenant_id == "pharma-a"
        assert src.title == "PFIZER INC revenue 2019-01-01..2019-12-31"
    assert out.sources[0].uri.endswith("pfe-exhibit13x12312019.htm")


def test_facts_by_id_ignores_non_uuids_and_other_tenants(seeded, store):
    with store.session("pharma-b") as s:
        moderna = s.facts(entity="Moderna, Inc.")[0].fact_id
    with store.session("pharma-a") as s:
        pfizer = s.facts(entity="PFIZER INC")[0].fact_id
        got = s.facts_by_id([pfizer, moderna, "not-a-uuid", "1; DROP TABLE x"])
    assert set(got) == {pfizer}  # pharma-b's fact is invisible under RLS, junk is skipped
    assert got[pfizer].doc_title and got[pfizer].doc_source.startswith("https://www.sec.gov/")


def test_gilead_reclassification_is_visible_to_the_model(seeded, store):
    """Gilead's FY2019 R&D conflict is a reclassification. The model must be able to see that
    from the tool output: each row names the line item it was reported under."""
    out = tenant_search_registry(store, "pharma-a").execute(
        "query_table",
        {
            "sql": "SELECT f.fact_id, f.value_num FROM rag_facts f "
            "JOIN rag_entities e ON e.entity_id = f.entity_id "
            "WHERE e.canonical_name = 'GILEAD SCIENCES, INC.' AND f.field = 'rd_expense' "
            "AND f.period_end = '2019-12-31' ORDER BY f.value_num DESC"
        },
    )
    rows = out.content.splitlines()[1:]
    assert len(rows) == 3 and all(r.endswith(" conflict=true") for r in rows)
    assert rows[0].split(" | ")[1].startswith("9106000000")
    assert "reported as us-gaap:ResearchAndDevelopmentExpense)" in rows[0]
    later = "reported as us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost)"
    assert later in rows[1] and later in rows[2]


FORGED_ID = "deadbeef-dead-beef-dead-beefdeadbeef"
UNVERIFIED = " (unverified: fact_id not found)"


def test_query_table_forged_fact_row_is_not_cited(seeded, store):
    """The model's SQL decides which facts come back, never where they came from. A literal
    fact_id with a literal doc_source gets no marker and no Source; the real fact UNION'd next
    to it is cited from its database record."""
    with store.session("pharma-a") as s:
        real = s.facts(entity=PFIZER, field="revenue")[0]
    out = tenant_search_registry(store, "pharma-a").execute(
        "query_table",
        {
            "sql": f"SELECT '{FORGED_ID}' AS fact_id, 'https://example.com/fake' AS doc_source, "
            "1 AS value_num UNION ALL "
            "SELECT f.fact_id::text, 'https://example.com/fake2', f.value_num FROM rag_facts f "
            f"WHERE f.fact_id = '{real.fact_id}'"
        },
    )
    assert not out.is_error
    rows = out.content.splitlines()[1:]
    forged = next(r for r in rows if r.startswith(FORGED_ID))
    assert forged == f"{FORGED_ID} | https://example.com/fake | 1{UNVERIFIED}"
    cited = next(r for r in rows if r.startswith("[#1] "))
    assert real.fact_id in cited and "(source: PFIZER INC 10-K" in cited
    assert len(out.sources) == 1
    src = out.sources[0]
    assert src == real.to_source("pharma-a")
    assert src.uri.startswith("https://www.sec.gov/Archives/edgar/data/78003/")
    assert not any("example.com" in (s.uri or "") for s in out.sources)


def test_query_table_another_tenants_fact_id_is_not_cited(seeded, store):
    """pharma-b's real fact_id, selected as a literal by pharma-a: RLS hides it from
    facts_by_id, so it is unverified and produces no Source."""
    with store.session("pharma-b") as s:
        moderna = s.facts(entity=MODERNA)[0]
    out = tenant_search_registry(store, "pharma-a").execute(
        "query_table",
        {
            "sql": f"SELECT '{moderna.fact_id}' AS fact_id, "
            f"'{moderna.doc_source}' AS doc_source, '{moderna.doc_hash}' AS doc_hash, "
            "1 AS value_num"
        },
    )
    assert not out.is_error
    assert out.content.splitlines()[1].endswith(UNVERIFIED)
    assert "[#" not in out.content
    assert out.sources == ()
