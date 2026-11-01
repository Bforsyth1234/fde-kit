"""Source adapters (PR 4): EDGAR over the committed offline fixtures, local files, and `ingest`
against a fake session. No network, no Postgres."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from fde_kit.rag.adapters import (
    FIELDS,
    AdapterBatch,
    EdgarAdapter,
    EdgarCacheMiss,
    EdgarClient,
    LocalFilesAdapter,
    RawDocument,
    RawEntity,
    RawFact,
    filings_for,
    html_to_text,
    infer_precision,
    ingest,
    load_manifest,
)
from fde_kit.rag.adapters.edgar import parse_filing_index
from fde_kit.rag.store import FactRow, IngestOutcome

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evals" / "fixtures" / "edgar"
CACHE = FIXTURES / "cache"
RFC = "RevenueFromContractWithCustomerExcludingAssessedTax"
PFE = ("0000078003-20-000014", "0000078003-21-000038", "0000078003-22-000027")


@pytest.fixture(scope="module")
def offline() -> EdgarClient:
    return EdgarClient("", cache_dir=CACHE, offline=True)


@pytest.fixture(scope="module")
def batches(offline: EdgarClient) -> dict[str, AdapterBatch]:
    filings = filings_for("pharma-a") + filings_for("pharma-b")
    out = {}
    for b in EdgarAdapter(offline, filings).batches():
        out[b.entities[0].aliases[0]] = b
    return out


def _fy2019(batch: AdapterBatch, field: str) -> list[RawFact]:
    return [
        f for f in batch.facts if f.row.field == field and f.row.period_end == date(2019, 12, 31)
    ]


# ---------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4162821000, 1000),
        (4162800000, 100000),
        (-747000000, 1000000),  # capped at 10**6
        (0, 1),
        (7, 1),
        (51750000000, 1000000),
        (-747064000, 1000),
    ],
)
def test_infer_precision(value: int, expected: int) -> None:
    assert infer_precision(value) == expected


def test_html_to_text_drops_hidden_and_joins_cells() -> None:
    html = """<html><head><title>Doc</title><style>p{color:red}</style></head><body>
    <div style="display:none"><ix:header><ix:hidden>dei:Secret 42</ix:hidden></ix:header></div>
    <script>var x = "nope";</script>
    <p>Total&nbsp;revenues   rose.</p><p>Second<br>line</p>
    <table><tr><td>Total revenues</td><td></td><td>$</td><td><span>51,750</span></td></tr>
    <tr><th>Net income</th><td><div>16,273</div></td></tr></table>
    <div>Tail &amp; end</div></body></html>"""
    text = html_to_text(html)
    assert text == (
        "Total revenues rose.\nSecond\nline\n"
        "Total revenues | $ | 51,750\nNet income | 16,273\nTail & end"
    )
    for hidden in ("Secret", "nope", "color", "Doc"):
        assert hidden not in text
    assert html_to_text(html) == text  # deterministic


def test_parse_filing_index_strips_inline_viewer_prefix() -> None:
    html = """<table class="tableFile" summary="Document Format Files">
    <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
    <tr><td>1</td><td>10-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/2/a.htm">a.htm</a>
    </td><td>10-K</td><td>9</td></tr>
    <tr><td>6</td><td>EXHIBIT 13</td><td><a href="/Archives/edgar/data/1/2/b.htm">b.htm</a></td>
    <td>EX-13</td><td>9</td></tr></table>
    <table class="tableFile" summary="Data Files"><tr><td>7</td><td>x</td>
    <td><a href="/Archives/edgar/data/1/2/c.xml">c.xml</a></td><td>EX-101.INS</td></tr></table>"""
    assert parse_filing_index(html) == [
        ("https://www.sec.gov/Archives/edgar/data/1/2/a.htm", "10-K"),
        ("https://www.sec.gov/Archives/edgar/data/1/2/b.htm", "EX-13"),
    ]


# ---------------------------------------------------------------------- client


def test_client_requires_user_agent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FDE_EDGAR_USER_AGENT"):
        EdgarClient("", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        EdgarClient("   ", cache_dir=tmp_path)


def test_offline_client_raises_on_miss(tmp_path: Path) -> None:
    client = EdgarClient("test-agent contact", cache_dir=tmp_path, offline=True)
    with pytest.raises(EdgarCacheMiss, match="not in the cache"):
        client.json("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json")
    with pytest.raises(EdgarCacheMiss):
        client.text("https://www.sec.gov/Archives/nothing.htm")
    assert client.requests == 0
    assert "test-agent" not in repr(client)


def test_cache_entries_are_url_keyed(offline: EdgarClient) -> None:
    for path in CACHE.iterdir():
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert set(entry) <= {"url", "body", "extracted"}
        assert path.stem == hashlib.sha1(entry["url"].encode()).hexdigest()
        assert path.suffix in (".json", ".txt")


# ---------------------------------------------------------------------- EDGAR adapter


def test_edgar_offline_yields_five_company_batches(batches: dict[str, AdapterBatch]) -> None:
    assert sorted(batches) == ["GILD", "MRNA", "PFE", "REGN", "VRTX"]
    pfe = batches["PFE"].entities
    assert pfe == [RawEntity("company", "PFIZER INC", ("PFE", "78003"))]
    assert batches["MRNA"].entities[0].aliases == ("MRNA", "1682852")
    for b in batches.values():
        assert {d.field for d in b.dictionary} == set(FIELDS)
        assert {f.row.field for f in b.facts} == set(FIELDS)
        sources = {d.source for d in b.documents}
        assert all(f.document_source in sources for f in b.facts)
        assert all(len(d.text) <= 40_000 for d in b.documents)


def test_pfizer_documents_and_ex13_citation(batches: dict[str, AdapterBatch]) -> None:
    docs = batches["PFE"].documents
    # FY2019's 10-K is a shell incorporating the financials from EX-13; from FY2020 on
    # Pfizer files the financial statements in the 10-K itself (no EX-13).
    assert [(d.metadata["fiscal_year"], d.metadata["doc_type"]) for d in docs] == [
        ("2019", "10-K"),
        ("2019", "EX-13"),
        ("2020", "10-K"),
        ("2021", "10-K"),
    ]
    ex13 = docs[1]
    assert ex13.source.endswith("/000007800320000014/pfe-exhibit13x12312019.htm")
    assert ex13.title == "PFIZER INC 10-K FY2019 (EX-13)"
    assert ex13.metadata["accession"] == PFE[0]
    by_accn = {
        f.row.source_ref.rsplit("@", 1)[1]: f.document_source
        for f in _fy2019(batches["PFE"], "revenue")
    }
    assert by_accn[PFE[0]] == ex13.source
    assert by_accn[PFE[1]] == docs[2].source
    assert by_accn[PFE[2]] == docs[3].source


def test_pfizer_fy2019_revenue_three_ways(batches: dict[str, AdapterBatch]) -> None:
    rows = [f.row for f in _fy2019(batches["PFE"], "revenue")]
    assert [r.value for r in rows] == [51750000000, 41172000000, 40905000000]
    assert [r.source_ref for r in rows] == [f"us-gaap:{RFC}@{a}" for a in PFE]
    for r in rows:
        assert isinstance(r, FactRow)
        assert r.precision == 1_000_000
        assert r.entity == "PFIZER INC" and r.unit == "USD"
        assert r.period_start == date(2019, 1, 1) and r.doc_id is None


def test_moderna_tag_is_chosen_per_filing(batches: dict[str, AdapterBatch]) -> None:
    refs = [f.row.source_ref.split("@")[0] for f in _fy2019(batches["MRNA"], "revenue")]
    assert refs == [f"us-gaap:{RFC}", "us-gaap:Revenues", "us-gaap:Revenues"]


def test_vertex_precision_only_difference(batches: dict[str, AdapterBatch]) -> None:
    rows = [f.row for f in _fy2019(batches["VRTX"], "revenue")]
    assert [(r.value, r.precision) for r in rows] == [
        (4162821000, 1000),
        (4162821000, 1000),
        (4162800000, 100000),
    ]


def test_full_year_periods_only(batches: dict[str, AdapterBatch]) -> None:
    for b in batches.values():
        for f in b.facts:
            days = (f.row.period_end - f.row.period_start).days
            assert 350 <= days <= 372
            assert 2019 <= f.row.period_end.year <= 2021


# ---------------------------------------------------------------------- manifest


def test_manifest_pins_fifteen_filings() -> None:
    entries = load_manifest()
    assert len(entries) == 15
    assert len({e.accession for e in entries}) == 15
    for e in entries:
        assert e.form == "10-K" and e.fiscal_year in (2019, 2020, 2021)
        assert e.documents and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in e.documents.values())
    b = filings_for("pharma-b")
    assert {f.ticker for f in b} == {"MRNA", "REGN"} and len(b) == 6
    assert {f.ticker for f in filings_for("pharma-a")} == {"PFE", "GILD", "VRTX"}


def test_fixtures_hold_no_contact_strings() -> None:
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    paths = [*FIXTURES.rglob("*"), ROOT / "src/fde_kit/rag/adapters/edgar_manifest.json"]
    for path in paths:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not email.search(text), path  # no contact address of any kind


# ---------------------------------------------------------------------- ingest


@dataclass
class _F:
    conflict_group_id: str | None


class FakeSession:
    """Versions by hash like `TenantSession`: a first version is `doc:<source>`, a new hash is
    `doc:<source>@v<n>` with the superseded doc_id as `previous_doc_id`."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.hashes: dict[str, str] = {}
        self.doc_ids: dict[str, str] = {}
        self.versions: dict[str, int] = {}
        self.facts_added: list[FactRow] = []
        self.jobs: list[tuple[str, str, str]] = []

    def upsert_entity(self, kind, canonical_name, aliases=()):
        self.calls.append(("upsert_entity", kind, canonical_name, tuple(aliases)))
        return f"e:{canonical_name}"

    def ingest_versioned(self, source, text, *, title="", metadata=None, max_chars=900):
        digest = hashlib.sha256(text.encode()).hexdigest()
        self.calls.append(("ingest_versioned", source, max_chars))
        previous = self.doc_ids.get(source)
        if self.hashes.get(source) == digest:
            return IngestOutcome(previous or "", self.versions[source], False, digest)
        version = self.versions.get(source, 0) + 1
        doc_id = f"doc:{source}" if version == 1 else f"doc:{source}@v{version}"
        self.hashes[source], self.doc_ids[source], self.versions[source] = digest, doc_id, version
        return IngestOutcome(doc_id, version, True, digest, previous)

    def start_extraction_job(self, kind, *, model="", prompt_sha256=""):
        self.calls.append(("start_extraction_job", kind))
        self.jobs.append((kind, model, prompt_sha256))
        return "job-1"

    def add_facts(self, rows, *, extraction_job_id=None):
        self.calls.append(("add_facts", len(rows), extraction_job_id))
        self.facts_added.extend(rows)
        return [f"f{i}" for i in range(len(rows))]

    def delete_facts_for_document(self, doc_id):
        self.calls.append(("delete_facts_for_document", doc_id))
        n = sum(1 for r in self.facts_added if r.doc_id == doc_id)
        self.facts_added = [r for r in self.facts_added if r.doc_id != doc_id]
        return n

    def set_dictionary(self, entries):
        self.calls.append(("set_dictionary", len(list(entries))))

    def facts(self, entity=None, field=None):
        return [_F("g1"), _F(None), _F("g1")] if entity == "Acme" else []


class _ListAdapter:
    name = "fake"

    def __init__(self, batches: list[AdapterBatch]):
        self._batches = batches

    def batches(self):
        yield from self._batches


def _batch(doc_source: str = "doc-a") -> AdapterBatch:
    row = FactRow(
        entity="Acme", field="revenue", value=100, unit="USD", precision=1, source_ref="x@1"
    )
    return AdapterBatch(
        entities=[RawEntity("company", "Acme", ("ACME",))],
        documents=[RawDocument("doc-a", "A", "alpha"), RawDocument("doc-b", "B", "beta")],
        facts=[RawFact(doc_source, row)],
    )


def test_ingest_is_idempotent_and_fills_doc_id() -> None:
    s = FakeSession()
    adapter = _ListAdapter([_batch()])
    first = ingest(s, adapter, max_chars=500)
    assert (first.documents, first.documents_changed, first.facts) == (2, 2, 1)
    assert first.entities == 1 and first.conflict_groups == 1 and first.job_id == "job-1"
    assert s.facts_added[0].doc_id == "doc:doc-a"
    assert s.facts_added[0].precision == 1 and s.facts_added[0].source_ref == "x@1"
    assert ("start_extraction_job", "adapter:fake") in s.calls
    assert ("ingest_versioned", "doc-a", 500) in s.calls

    second = ingest(s, adapter)
    assert (second.documents, second.documents_changed, second.facts) == (2, 0, 0)
    assert second.job_id == ""
    assert len(s.facts_added) == 1
    assert sum(1 for c in s.calls if c[0] == "add_facts") == 1
    assert sum(1 for c in s.calls if c[0] == "start_extraction_job") == 1


def _export(text: str, values: tuple[int, ...]) -> AdapterBatch:
    """One sheet-like document re-exported with new content, citing `values` as facts."""
    rows = [
        FactRow(entity="Acme", field=f"m{i}", value=v, unit="USD", source_ref=f"Plan!B{i}")
        for i, v in enumerate(values, start=2)
    ]
    return AdapterBatch(
        entities=[RawEntity("company", "Acme")],
        documents=[RawDocument("plan.xlsx#Plan", "Plan", text)],
        facts=[RawFact("plan.xlsx#Plan", r) for r in rows],
    )


def test_fake_session_previous_doc_id() -> None:
    s = FakeSession()
    first = s.ingest_versioned("a", "one")
    assert (first.doc_id, first.version, first.changed, first.previous_doc_id) == (
        "doc:a",
        1,
        True,
        None,
    )
    same = s.ingest_versioned("a", "one")
    assert (same.doc_id, same.changed, same.previous_doc_id) == ("doc:a", False, None)
    second = s.ingest_versioned("a", "two")
    assert (second.doc_id, second.version, second.previous_doc_id) == ("doc:a@v2", 2, "doc:a")


def test_ingest_keep_leaves_superseded_facts() -> None:
    s = FakeSession()
    ingest(s, _ListAdapter([_export("v1", (1, 2, 3))]))
    report = ingest(s, _ListAdapter([_export("v2", (1, 2, 4))]))
    assert (report.documents_changed, report.facts, report.facts_replaced) == (1, 3, 0)
    assert not any(c[0] == "delete_facts_for_document" for c in s.calls)
    assert [r.doc_id for r in s.facts_added] == ["doc:plan.xlsx#Plan"] * 3 + [
        "doc:plan.xlsx#Plan@v2"
    ] * 3


def test_ingest_replace_deletes_superseded_facts_before_adding() -> None:
    s = FakeSession()
    first = ingest(s, _ListAdapter([_export("v1", (1, 2, 3))]), on_new_version="replace")
    assert first.facts_replaced == 0  # a first version supersedes nothing
    assert not any(c[0] == "delete_facts_for_document" for c in s.calls)

    s.calls.clear()
    report = ingest(s, _ListAdapter([_export("v2", (1, 2, 4))]), on_new_version="replace")
    assert (report.documents_changed, report.facts, report.facts_replaced) == (1, 3, 3)
    names = [c[0] for c in s.calls]
    assert names.index("delete_facts_for_document") < names.index("add_facts")
    assert ("delete_facts_for_document", "doc:plan.xlsx#Plan") in s.calls
    assert [(r.doc_id, r.value) for r in s.facts_added] == [
        ("doc:plan.xlsx#Plan@v2", 1),
        ("doc:plan.xlsx#Plan@v2", 2),
        ("doc:plan.xlsx#Plan@v2", 4),
    ]

    s.calls.clear()  # unchanged re-ingest: nothing deleted, nothing written
    again = ingest(s, _ListAdapter([_export("v2", (1, 2, 4))]), on_new_version="replace")
    assert (again.documents_changed, again.facts, again.facts_replaced) == (0, 0, 0)
    assert not any(c[0] in ("delete_facts_for_document", "add_facts") for c in s.calls)


def test_ingest_rejects_unknown_on_new_version() -> None:
    s = FakeSession()
    with pytest.raises(ValueError, match="on_new_version"):
        ingest(s, _ListAdapter([_batch()]), on_new_version="purge")  # type: ignore[arg-type]
    assert s.calls == []


def test_ingest_job_kind_from_adapter_attributes() -> None:
    s = FakeSession()
    ingest(s, _ListAdapter([_batch()]))
    assert s.jobs == [("adapter:fake", "", "")]

    adapter = _ListAdapter([_batch()])
    adapter.job_kind = "wrangle:plan@0123456789ab"  # type: ignore[attr-defined]
    adapter.job_model = "none"  # type: ignore[attr-defined]
    adapter.job_prompt_sha256 = "f" * 64  # type: ignore[attr-defined]
    s = FakeSession()
    ingest(s, adapter)
    assert s.jobs == [("wrangle:plan@0123456789ab", "none", "f" * 64)]


def test_ingest_rejects_fact_citing_unknown_document() -> None:
    s = FakeSession()
    with pytest.raises(ValueError, match="not in their batch"):
        ingest(s, _ListAdapter([_batch("doc-zzz")]))
    assert not any(c[0] == "ingest_versioned" for c in s.calls)


def test_ingest_edgar_offline_against_fake_session(offline: EdgarClient) -> None:
    s = FakeSession()
    report = ingest(s, EdgarAdapter(offline, filings_for("pharma-b")))
    assert report.documents == report.documents_changed == 6
    assert report.facts == len(s.facts_added) == 36
    assert all(r.doc_id and r.doc_id.startswith("doc:https://www.sec.gov/") for r in s.facts_added)
    assert ingest(s, EdgarAdapter(offline, filings_for("pharma-b"))).facts == 0


# ---------------------------------------------------------------------- local files


def test_local_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# Lease\n\nRent is $1,800.", encoding="utf-8")
    (tmp_path / "b.html").write_text(
        "<html><body><p>Hello</p><table><tr><td>Rent</td><td>1,850</td></tr></table>"
        "<script>x()</script></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 not really")
    (tmp_path / "facts.jsonl").write_text(
        json.dumps(
            {
                "document": "notes/a.md",
                "entity": "Unit 4B",
                "field": "rent",
                "value": 1800,
                "unit": "USD",
                "period_start": "2026-01-01",
                "period_end": None,
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    adapter = LocalFilesAdapter(tmp_path, entity="Acme Properties")
    with caplog.at_level(logging.WARNING):
        (batch,) = list(adapter.batches())
    assert "scan.pdf" in caplog.text and adapter.skipped == ["scan.pdf"]
    assert [d.source for d in batch.documents] == ["b.html", "notes/a.md"]
    assert batch.documents[0].text == "Hello\nRent | 1,850"
    assert batch.documents[1].text.startswith("# Lease")
    assert [e.canonical_name for e in batch.entities] == ["Acme Properties", "Unit 4B"]
    (fact,) = batch.facts
    assert fact.document_source == "notes/a.md"
    assert fact.row.value == 1800 and fact.row.period_start == date(2026, 1, 1)
    assert fact.row.period_end is None and fact.row.precision is None

    s = FakeSession()
    report = ingest(s, adapter)
    assert report.documents == 2 and report.facts == 1
    assert s.facts_added[0].doc_id == "doc:notes/a.md"


def test_local_files_without_sidecar(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("plain", encoding="utf-8")
    (batch,) = list(LocalFilesAdapter(tmp_path).batches())
    assert batch.facts == [] and batch.entities == []
    assert [d.source for d in batch.documents] == ["a.txt"]
