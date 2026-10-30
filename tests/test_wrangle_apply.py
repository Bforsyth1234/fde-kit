"""Apply (§7.4), gates (§8) and the adapter (§7.5), against the fixture oracle.

The expected facts and rejects were written from the fixture generator's world, not by running
this code, so matching them is the real check. The provenance read-back test re-derives every
stored value from the cell its `source_ref` names.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from fde_kit.rag.adapters.base import AdapterBatch, ingest
from fde_kit.wrangle.adapter import WrangleAdapter, apply_to_tenant, job_kind_for
from fde_kit.wrangle.apply import (
    ApplyResult,
    SpecMismatch,
    cell_pipeline,
    gate,
    parse_source_ref,
    run,
    write_outputs,
)
from fde_kit.wrangle.labels import header_fingerprint, header_labels
from fde_kit.wrangle.parse import to_fact_value
from fde_kit.wrangle.read import parse_ref, read_grid
from fde_kit.wrangle.spec import LoadedSpec, approve, load_project_spec
from tests.test_adapters import FakeSession

FIXTURES = Path(__file__).parent / "fixtures" / "wrangle"
PROJECT = FIXTURES / "project"
SOURCES = {
    "erp_shipments": "erp_shipments.csv",
    "planner_forecast": "planner_forecast.xlsx",
    "legacy_report": "legacy_report.csv",
    "formulas": "formulas.xlsx",
}


# §7.6 fixes the read-failure wording; the fixture's advisory reason predates it.
def _spec(name: str) -> Path:
    return PROJECT / "specs" / f"{name}.yaml"


def _run(name: str) -> ApplyResult:
    return run(_spec(name), [FIXTURES / SOURCES[name]])


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def results() -> dict[str, ApplyResult]:
    return {name: _run(name) for name in SOURCES}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A writable copy of the fixture project, plus the source files beside it."""
    dst = tmp_path / "project"
    shutil.copytree(PROJECT, dst)
    for f in SOURCES.values():
        shutil.copy(FIXTURES / f, tmp_path / f)
    return dst


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --- the oracle -----------------------------------------------------------------------------


def _fact_key(d: dict) -> tuple:
    value = d["value"]
    return (
        d["entity"],
        d["field"],
        d["unit"],
        d["period_start"],
        d["period_end"],
        d["precision"],
        d["source_ref"].split("@")[0],
        type(value) is str,
    )


@pytest.mark.parametrize("name", list(SOURCES))
def test_facts_match_expected(name: str, results: dict[str, ApplyResult]) -> None:
    result = results[name]
    got = {}
    for f in result.facts:
        row = f.row
        d = {
            "entity": row.entity,
            "field": row.field,
            "value": row.value,
            "unit": row.unit,
            "period_start": row.period_start.isoformat(),
            "period_end": row.period_end.isoformat(),
            "precision": row.precision,
            "source_ref": row.source_ref,
        }
        assert _fact_key(d) not in got, f"two facts for {d}"
        got[_fact_key(d)] = row.value
    want = {_fact_key(d): d["value"] for d in _jsonl(FIXTURES / "expected" / f"{name}.facts.jsonl")}
    assert len(got) == len(result.facts)
    assert sorted(got) == sorted(want)
    for k, v in want.items():
        if isinstance(v, str):
            assert got[k] == v, k
        else:
            assert math.isclose(got[k], v, rel_tol=0, abs_tol=1e-9), (k, got[k], v)
            assert isinstance(got[k], int) == isinstance(v, int), (k, got[k], v)


@pytest.mark.parametrize("name", list(SOURCES))
def test_rejects_match_expected(name: str, results: dict[str, ApplyResult]) -> None:
    def where(r: dict) -> tuple:
        return (r["sheet"], r.get("cell"), r.get("row"), r["level"], r["stage"])

    got = {where(r): r for r in results[name].rejects}
    want = {where(r): r for r in _jsonl(FIXTURES / "expected" / f"{name}.rejects.jsonl")}
    assert sorted(got, key=str) == sorted(want, key=str)
    for k, w in want.items():
        g = got[k]
        assert Path(g["file"]).name == w["file"]
        for opt in ("raw", "op"):
            if opt in w:
                assert g[opt] == w[opt], (k, opt)
        assert g["reason"] == w["reason"]


def test_reject_records_have_the_frozen_shape(results: dict[str, ApplyResult]) -> None:
    for result in results.values():
        for r in result.rejects:
            where = "cell" if r["level"] == "cell" else "row"
            assert list(r) == ["file", "sheet", where, "level", "raw", "stage", "op", "reason"]
            assert r["stage"] in ("entity", "period", "measure")


def test_formula_cells_are_rejects_never_zero(results: dict[str, ApplyResult]) -> None:
    rejects = {r["cell"]: r for r in results["formulas"].rejects}
    assert rejects["D4"]["op"] == "read" and rejects["D4"]["raw"] == ""
    assert rejects["D4"]["reason"] == "formula without cached value"
    assert rejects["D6"]["op"] == "read" and rejects["D6"]["reason"] == "error cell"
    refs = {f.row.source_ref.split("@")[0] for f in results["formulas"].facts}
    assert refs == {"Calc!D3", "Calc!D5"}


REPORTS = {
    "erp_shipments": dict(
        rows_read=61,
        rows_skipped={},
        empty_skipped=0,
        facts=59,
        duplicates_dropped=1,
        rejects=1,
        entities={"exact": 5, "normalised": 0, "fuzzy": 0, "new": 0},
        by_reason={"number: unparseable": 1},
    ),
    "planner_forecast": dict(
        rows_read=15,
        rows_skipped={"skip_rows[0]": 6},
        empty_skipped=1,
        facts=35,
        duplicates_dropped=0,
        rejects=0,
        # Sites: 4 exact + TOLEDO PLT 12 by norm_key; parts: 3 open values.
        entities={"exact": 4, "normalised": 1, "fuzzy": 0, "new": 3},
        by_reason={},
    ),
    "legacy_report": dict(
        rows_read=33,
        rows_skipped={},
        empty_skipped=0,
        facts=96,
        duplicates_dropped=0,
        rejects=1,
        entities={"exact": 5, "normalised": 0, "fuzzy": 0, "new": 0},
        by_reason={"unknown entity": 1},
    ),
    "formulas": dict(
        rows_read=5,
        rows_skipped={"skip_rows[0]": 1},
        empty_skipped=0,
        facts=2,
        duplicates_dropped=0,
        rejects=2,
        entities={"exact": 0, "normalised": 0, "fuzzy": 0, "new": 4},
        by_reason={"error cell": 1, "formula without cached value": 1},
    ),
}


@pytest.mark.parametrize("name", list(SOURCES))
def test_report(name: str, results: dict[str, ApplyResult]) -> None:
    result, want = results[name], REPORTS[name]
    rep = result.report
    assert list(rep) == [
        "spec",
        "bundle_sha256",
        "approved",
        "files",
        "facts",
        "duplicates_dropped",
        "rejects",
        "entities",
        "conflicts_within_load",
        "similar_entities",
    ]
    assert rep["spec"] == name and rep["bundle_sha256"] == result.loaded.bundle_sha256
    assert rep["approved"] is False
    (f,) = rep["files"]
    path = FIXTURES / SOURCES[name]
    assert f["path"] == str(path)
    assert f["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert f["sheet"] == ({"planner_forecast": "Plan", "formulas": "Calc"}.get(name, path.stem))
    assert f["rows_read"] == want["rows_read"]
    assert f["rows_skipped"] == want["rows_skipped"]
    assert f["empty_skipped"] == want["empty_skipped"]
    assert rep["facts"] == want["facts"]
    assert rep["duplicates_dropped"] == want["duplicates_dropped"]
    assert rep["rejects"]["total"] == want["rejects"]
    assert rep["rejects"]["by_reason"] == want["by_reason"]
    rate = want["rejects"] / (want["facts"] + want["rejects"])
    assert rep["rejects"]["rate"] == pytest.approx(rate, abs=1e-6)
    assert result.reject_rate == pytest.approx(rate)
    assert rep["entities"] == want["entities"]
    assert rep["conflicts_within_load"] == []
    json.dumps(rep)  # serialisable as is


def test_reject_rates_from_fixture_readme(results: dict[str, ApplyResult]) -> None:
    assert results["erp_shipments"].reject_rate == pytest.approx(1 / 60)
    assert results["legacy_report"].reject_rate == pytest.approx(1 / 97)
    assert results["planner_forecast"].reject_rate == 0


def test_duplicate_keeps_first_row(results: dict[str, ApplyResult]) -> None:
    saltillo_feb = [
        f.row
        for f in results["erp_shipments"].facts
        if f.row.entity == "Saltillo Castings 21" and f.row.period_start == date(2026, 2, 1)
    ]
    assert [r.source_ref.split("@")[0] for r in saltillo_feb] == ["erp_shipments!C24"]


# --- provenance -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SOURCES))
def test_source_ref_full_form(name: str, results: dict[str, ApplyResult]) -> None:
    result = results[name]
    path = FIXTURES / SOURCES[name]
    file12 = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    suffix = f"@wrangle:{name}/{result.loaded.bundle_sha12}/{file12}"
    for f in result.facts:
        assert f.row.source_ref.endswith(suffix)
        assert f.row.source_ref.count("@") == 1


@pytest.mark.parametrize("name", list(SOURCES))
def test_provenance_read_back(name: str, results: dict[str, ApplyResult]) -> None:
    """Every fact: parse source_ref, open the file, read that cell, run the spec's ops and
    scale, and get the stored value back exactly."""
    result = results[name]
    loaded = load_project_spec(_spec(name))
    path = FIXTURES / SOURCES[name]
    measures = {m.field: m for m in loaded.spec.measures}
    # The same cell-aware pipeline apply uses (native XLSX numbers and dates).
    pipelines = {m.field: cell_pipeline(m.ops) for m in loaded.spec.measures if m.ops}
    grids: dict[str, object] = {}
    assert result.facts
    for f in result.facts:
        ref = parse_source_ref(f.row.source_ref)
        assert ref["spec"] == loaded.spec.spec
        assert ref["bundle"] == loaded.bundle_sha12
        if ref["sheet"] not in grids:
            sheet = ref["sheet"] if path.suffix == ".xlsx" else None
            grids[ref["sheet"]] = read_grid(
                path,
                sheet,
                encoding=loaded.spec.source.encoding,
                delimiter=loaded.spec.source.delimiter,
            )
        grid = grids[ref["sheet"]]
        assert grid.sheet == ref["sheet"]
        assert grid.file_sha256[:12] == ref["file"]
        cell = grid.cell(*parse_ref(ref["cell"]))
        m = measures[f.row.field]
        if m.is_text:
            expected = str(pipelines[m.field](cell)) if m.ops else cell.raw
        else:
            expected = to_fact_value(pipelines[m.field](cell) * m.scale_decimal)
        assert f.row.value == expected, (f.row.source_ref, cell.raw)
        assert type(f.row.value) is type(expected)


def test_parse_source_ref_rejects_other_refs() -> None:
    with pytest.raises(ValueError):
        parse_source_ref("us-gaap:Revenues@0000320193-24-000123")
    assert parse_source_ref("My Sheet!AB12@wrangle:x_1/0123456789ab/ba9876543210") == {
        "sheet": "My Sheet",
        "cell": "AB12",
        "spec": "x_1",
        "bundle": "0123456789ab",
        "file": "ba9876543210",
    }


# --- documents and entities -----------------------------------------------------------------


def test_document_per_sheet(results: dict[str, ApplyResult]) -> None:
    planner = results["planner_forecast"]
    (doc,) = planner.documents
    assert doc.source == "planner_forecast#Plan"
    assert doc.title == "planner_forecast.xlsx — Plan"
    first, body = doc.text.split("\n", 1)
    assert first == f"# wrangle spec=planner_forecast bundle={planner.loaded.bundle_sha256}"
    rows = list(csv.reader(io.StringIO(body)))
    assert rows[0] == ["row", "A", "B", "C", "D", "E", "F", "G"]
    assert rows[1][:2] == ["1", "Acme Components - S&OP Demand Plan"]  # title rows kept
    assert rows[2][:2] == ["2", "Units (000s)"]
    row14 = next(r for r in rows if r[0] == "14")
    assert row14[1:5] == ["Puebla Plt 33", "P-1234", "59.75", "1,012.5"]
    assert doc.metadata == {
        "file_sha256": planner.files[0].file_sha256,
        "spec": "planner_forecast",
        "bundle_sha256": planner.loaded.bundle_sha256,
        "header_rows": "[4, 5]",
    }
    assert all(isinstance(v, str) for v in doc.metadata.values())
    assert {f.document_source for f in planner.facts} == {doc.source}

    erp = results["erp_shipments"]
    (edoc,) = erp.documents
    # Default key "{spec}/{file}#{sheet}": the base name, however the path was typed.
    assert edoc.source == "erp_shipments/erp_shipments.csv#erp_shipments"
    assert edoc.title == "erp_shipments.csv — erp_shipments"
    erows = list(csv.reader(io.StringIO(edoc.text.split("\n", 1)[1])))
    assert erows[0] == ["row", "A", "B", "C"]
    assert erows[1] == ["1", "PLANT_CD", "SHIP_DT", "QTY_SHIPPED"]
    assert erows[28][0] == "28" and erows[28][3] == "38O37"
    assert len(erows) == 63


def test_entities(results: dict[str, ApplyResult]) -> None:
    planner = results["planner_forecast"]
    crosswalk = [e for e in planner.entities if e.kind == "plant"]
    assert len(crosswalk) == 5
    det = next(e for e in crosswalk if e.canonical_name == "Detroit Stamping 07")
    assert det.aliases == ("1007", "DET-07", "Detroit Plt 07")
    composite = [e for e in planner.entities if e.kind == "plant_part"]
    assert len(composite) == 9
    assert {e.canonical_name for e in composite} >= {
        "Detroit Stamping 07/P-1234",
        "Detroit Stamping 07/P-2210",
        "Toledo Assembly 12/P-3307",
        "Puebla Machining 33/P-3307",
    }
    assert all(e.aliases == () for e in composite)

    # A single-part crosswalk spec: its entities are the crosswalk's, nothing else.
    erp = results["erp_shipments"]
    assert [e.kind for e in erp.entities] == ["plant"] * 5
    # Open single-part entities: only those with facts (Nuts and Gaskets only rejected).
    formulas = results["formulas"]
    assert [(e.kind, e.canonical_name) for e in formulas.entities] == [
        ("item", "Bolts"),
        ("item", "Washers"),
    ]


def test_write_outputs(tmp_path: Path, results: dict[str, ApplyResult]) -> None:
    result = results["legacy_report"]
    paths = write_outputs(result, tmp_path / "out")
    assert json.loads(paths["report"].read_text()) == result.report
    assert _jsonl(paths["rejects"]) == result.rejects
    preview = _jsonl(paths["preview"])
    assert len(preview) == 50
    assert preview[0]["source_ref"] == result.facts[0].row.source_ref
    assert preview[0]["entity_cells"] == ["A2"] and preview[0]["period_from"] == "B2"
    assert paths["proposed"].read_text().splitlines() == [
        "crosswalk,alias,canonical,score,runner_up"
    ]


# --- spec mismatch --------------------------------------------------------------------------


def _fingerprint(path: Path, header_rows: list[int], sheet: str | None = None) -> str:
    return header_fingerprint(header_labels(read_grid(path, sheet), header_rows))


def test_fingerprint_match_and_mismatch(project: Path) -> None:
    spec = project / "specs" / "planner_forecast.yaml"
    xlsx = project.parent / "planner_forecast.xlsx"
    good = _fingerprint(xlsx, [4, 5], "Plan")
    _edit(spec, "  data_starts: 6\n", f"  data_starts: 6\n  header_fingerprint: {good}\n")
    assert run(spec, [xlsx]).report["facts"] == 35

    _edit(spec, f"header_fingerprint: {good}", "header_fingerprint: 0123456789ab")
    with pytest.raises(SpecMismatch, match="header_fingerprint"):
        run(spec, [xlsx])
    forced = run(spec, [xlsx], force=True)
    assert forced.report["facts"] == 35


def test_missing_column_is_spec_mismatch_even_with_force(project: Path) -> None:
    spec = project / "specs" / "erp_shipments.yaml"
    _edit(spec, "column: QTY_SHIPPED", "column: QTY_SHIPPD")
    csv_path = project.parent / "erp_shipments.csv"
    with pytest.raises(SpecMismatch, match="QTY_SHIPPD"):
        run(spec, [csv_path])
    with pytest.raises(SpecMismatch):
        run(spec, [csv_path], force=True)


def test_missing_unpivot_label_and_sheet(project: Path) -> None:
    spec = project / "specs" / "planner_forecast.yaml"
    xlsx = project.parent / "planner_forecast.xlsx"
    _edit(spec, '"Forecast / Feb-27"', '"Forecast / Mar-27"')
    with pytest.raises(SpecMismatch, match="Mar-27"):
        run(spec, [xlsx])
    _edit(spec, '"Forecast / Mar-27"', '"Forecast / Feb-27"')
    _edit(spec, "sheet: Plan", "sheet: Plann")
    with pytest.raises(SpecMismatch, match="Plann"):
        run(spec, [xlsx])


# --- row and cell level, duplicates and conflicts -------------------------------------------


def _erp_variant(project: Path, extra: str, name: str = "erp_shipments.csv") -> Path:
    path = project.parent / "variant" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes((FIXTURES / "erp_shipments.csv").read_bytes() + extra.encode())
    return path


def test_within_load_conflict_reported(project: Path) -> None:
    # Detroit Oct 2025 again (row 63) with a different quantity: both load, both cells named.
    path = _erp_variant(project, "1007,20251031,48700\n")
    result = run(project / "specs" / "erp_shipments.yaml", [path])
    assert result.report["facts"] == 60 and result.report["duplicates_dropped"] == 1
    assert result.report["conflicts_within_load"] == [
        {
            "document": "erp_shipments/erp_shipments.csv#erp_shipments",
            "key": ["Detroit Stamping 07", "shipped_units", "2025-10"],
            "cells": ["erp_shipments!C2", "erp_shipments!C63"],
        }
    ]
    det = sorted(
        f.row.value
        for f in result.facts
        if f.row.entity == "Detroit Stamping 07" and f.row.period_start == date(2025, 10, 1)
    )
    assert det == [48613, 48700]


def test_row_level_period_reject_and_empty_reject(project: Path) -> None:
    path = _erp_variant(project, "1007,2025-13-31,100\n1012,20261031,\n")
    spec = project / "specs" / "erp_shipments.yaml"
    result = run(spec, [path])
    period = [r for r in result.rejects if r["stage"] == "period"]
    assert period == [
        {
            "file": str(path),
            "sheet": "erp_shipments",
            "row": 63,
            "level": "row",
            "raw": "2025-13-31",
            "stage": "period",
            "op": "date",
            "reason": "date: unparseable",
        }
    ]
    assert result.report["files"][0]["empty_skipped"] == 1  # C64, empty: skip

    _edit(spec, "    empty: skip", "    empty: reject")
    rejected = run(spec, [path])
    empty = [r for r in rejected.rejects if r.get("cell") == "C64"]
    assert len(empty) == 1 and empty[0]["level"] == "cell" and empty[0]["stage"] == "measure"
    assert rejected.report["files"][0]["empty_skipped"] == 0


def test_two_files_in_one_load_dedupe_per_document(project: Path) -> None:
    """Duplicates are dropped (and conflicts reported) within a document only. Across
    documents every fact loads, and `add_facts` decides agreement or conflict at ingest, as
    it would between two loads: dropping the copy's facts would leave its document with
    none, and with a fixed key a later file's facts would be retired with nothing in their
    place."""
    a = FIXTURES / "erp_shipments.csv"
    b = _erp_variant(project, "", name="erp_copy.csv")
    result = run(project / "specs" / "erp_shipments.yaml", [a, b])
    assert [len(f.facts) for f in result.files] == [59, 59]
    assert result.report["facts"] == 118
    assert result.report["duplicates_dropped"] == 1 + 1  # Saltillo Feb 2026, in each file
    assert result.report["conflicts_within_load"] == []
    assert len(result.rejects) == 2


def test_two_files_with_one_document_key_are_refused(project: Path) -> None:
    """Two planner exports under its fixed key, in one load: the second would supersede the
    first within the load (and retire its facts), so the load is refused before any read."""
    import openpyxl

    spec = project / "specs" / "planner_forecast.yaml"
    oct_ = project.parent / "planner_forecast.xlsx"
    wb = openpyxl.load_workbook(oct_)
    wb["Plan"]["D14"] = 999.0
    nov = project.parent / "Forecast_Nov.xlsx"
    wb.save(nov)
    with pytest.raises(SpecMismatch, match="supersede each other") as e:
        run(spec, [oct_, nov])
    assert "planner_forecast#Plan" in str(e.value) and str(oct_) in str(e.value)

    # The same file twice under the default key, too, however the path is typed.
    erp = project.parent / "erp_shipments.csv"
    with pytest.raises(SpecMismatch, match="supersede each other"):
        run(project / "specs" / "erp_shipments.yaml", [erp, f"{erp.parent}/./{erp.name}"])


# --- gates ----------------------------------------------------------------------------------


def test_gate_unapproved_and_draft(results: dict[str, ApplyResult]) -> None:
    erp = results["erp_shipments"]
    reasons = gate(erp, approved=erp.approved, draft=False, max_reject_rate=0.02)
    assert len(reasons) == 1 and "not approved" in reasons[0]
    assert gate(erp, approved=False, draft=True, max_reject_rate=0.02) == []
    over = gate(erp, approved=False, draft=True, max_reject_rate=0.01)
    assert len(over) == 1 and "reject rate" in over[0]


def test_gate_approved_stale_and_rate(project: Path) -> None:
    spec = project / "specs" / "erp_shipments.yaml"
    csv_path = project.parent / "erp_shipments.csv"
    approve(spec, by="tester")
    ok = run(spec, [csv_path])
    assert ok.approved and ok.report["approved"] is True
    assert gate(ok, approved=True, draft=False, max_reject_rate=0.02) == []
    assert gate(ok) == []
    # Exactly at the limit is allowed; just under it refuses.
    assert gate(ok, max_reject_rate=1 / 60) == []
    assert gate(ok, max_reject_rate=0.016)

    # One crosswalk byte changes: the approval goes stale.
    cw = project / "crosswalks" / "plants.csv"
    cw.write_bytes(cw.read_bytes() + b"\n")
    stale = run(spec, [csv_path])
    assert not stale.approved
    reasons = gate(stale, approved=stale.approved, draft=False, max_reject_rate=0.02)
    assert len(reasons) == 1 and "stale" in reasons[0]
    assert gate(stale, draft=True) == []


def test_gate_refuses_formulas_rate(project: Path) -> None:
    spec = project / "specs" / "formulas.yaml"
    approve(spec, by="tester")
    result = run(spec, [project.parent / "formulas.xlsx"])
    reasons = gate(result, approved=result.approved, draft=False, max_reject_rate=0.02)
    assert len(reasons) == 1 and "0.5000" in reasons[0]


# --- adapter and ingest ---------------------------------------------------------------------


class RecordingSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.documents: dict[str, tuple[str, str, dict]] = {}

    def ingest_versioned(self, source, text, *, title="", metadata=None, max_chars=900):
        self.documents[source] = (title, text, dict(metadata or {}))
        return super().ingest_versioned(
            source, text, title=title, metadata=metadata, max_chars=max_chars
        )


def _upserts(s: FakeSession) -> dict[tuple[str, str], tuple[str, ...]]:
    return {(c[1], c[2]): c[3] for c in s.calls if c[0] == "upsert_entity"}


def _approved(project: Path, name: str) -> LoadedSpec:
    return approve(project / "specs" / f"{name}.yaml", by="tester")


def test_adapter_shape(results: dict[str, ApplyResult]) -> None:
    result = results["planner_forecast"]
    adapter = WrangleAdapter.from_result(result)
    assert adapter.name == "wrangle"
    assert adapter.job_kind == f"wrangle-draft:planner_forecast@{result.loaded.bundle_sha12}"
    assert adapter.job_model == "" and adapter.job_prompt_sha256 == ""
    (batch,) = list(adapter.batches())
    assert isinstance(batch, AdapterBatch)
    assert len(batch.entities) == 5 + 9
    assert batch.documents == result.documents
    assert len(batch.facts) == 35
    assert [d.field for d in batch.dictionary] == ["demand_forecast_units"]


def test_adapter_from_spec_and_files_one_batch_per_file(project: Path) -> None:
    variant = _erp_variant(project, "", name="erp_copy.csv")
    adapter = WrangleAdapter(
        project / "specs" / "erp_shipments.yaml", [FIXTURES / "erp_shipments.csv", variant]
    )
    batches = list(adapter.batches())
    assert len(batches) == 2
    assert [len(b.documents) for b in batches] == [1, 1]
    assert [len(b.facts) for b in batches] == [59, 59]


def test_apply_to_tenant_approved(project: Path) -> None:
    loaded = _approved(project, "planner_forecast")
    xlsx = project.parent / "planner_forecast.xlsx"
    s = RecordingSession()
    result, report = apply_to_tenant(s, loaded, [xlsx])
    assert report is not None
    kind = f"wrangle:planner_forecast@{loaded.bundle_sha12}"
    assert job_kind_for(loaded) == kind
    assert s.jobs == [(kind, "", "")]
    assert (report.documents, report.documents_changed, report.facts) == (1, 1, 35)
    assert report.facts_replaced == 0

    up = _upserts(s)
    assert up[("plant", "Toledo Assembly 12")] == ("1012", "TOL-12", "Toledo Plt 12")
    assert up[("plant_part", "Toledo Assembly 12/P-3307")] == ()
    assert len([k for k in up if k[0] == "plant"]) == 5
    assert len([k for k in up if k[0] == "plant_part"]) == 9

    title, text, meta = s.documents["planner_forecast#Plan"]
    assert title == "planner_forecast.xlsx — Plan"
    head = f"# wrangle spec=planner_forecast bundle={loaded.bundle_sha256}\n"
    assert text.startswith(head + "row,A,B,")
    assert meta["spec"] == "planner_forecast" and meta["bundle_sha256"] == loaded.bundle_sha256
    assert len(s.facts_added) == 35
    assert {r.doc_id for r in s.facts_added} == {"doc:planner_forecast#Plan"}
    assert ("set_dictionary", 1) in s.calls

    # Unchanged re-run: nothing written, no job.
    _, again = apply_to_tenant(s, loaded, [xlsx])
    assert again is not None and (again.documents_changed, again.facts, again.job_id) == (0, 0, "")
    assert len(s.jobs) == 1


def test_reexport_with_one_changed_cell_replaces_facts(project: Path) -> None:
    import openpyxl

    loaded = _approved(project, "planner_forecast")
    first_xlsx = project.parent / "planner_forecast.xlsx"
    s = RecordingSession()
    _, first = apply_to_tenant(s, loaded, [first_xlsx])
    assert first is not None and first.facts == 35
    before = {r.source_ref.split("@")[0]: r.value for r in s.facts_added}

    # Next month's export, one cell changed, saved under another name: the fixed
    # document_key makes it the next version of the same document.
    wb = openpyxl.load_workbook(first_xlsx)
    wb["Plan"]["D6"] = 14.5
    second_xlsx = project.parent / "planner_forecast_nov.xlsx"
    wb.save(second_xlsx)
    result, second = apply_to_tenant(s, loaded, [second_xlsx])
    assert second is not None
    assert (second.documents_changed, second.facts, second.facts_replaced) == (1, 35, 35)
    assert ("delete_facts_for_document", "doc:planner_forecast#Plan") in s.calls
    assert len(s.facts_added) == 35
    assert {r.doc_id for r in s.facts_added} == {"doc:planner_forecast#Plan@v2"}
    after = {r.source_ref.split("@")[0]: r.value for r in s.facts_added}
    diff = {k for k in before if before[k] != after[k]}
    assert diff == {"Plan!D6"} and after["Plan!D6"] == 14500
    assert all(r.source_ref.endswith(result.files[0].file_sha256[:12]) for r in s.facts_added)


def test_apply_to_tenant_refused_writes_nothing(project: Path) -> None:
    s = RecordingSession()
    spec = project / "specs" / "erp_shipments.yaml"
    csv_path = project.parent / "erp_shipments.csv"
    result, report = apply_to_tenant(s, load_project_spec(spec), [csv_path])
    assert report is None and s.calls == []
    assert result.report["facts"] == 59  # the dry run still says what it would have done

    # Rate over the limit, even when approved.
    approve(spec, by="tester")
    _, report = apply_to_tenant(s, spec, [csv_path], max_reject_rate=0.01)
    assert report is None and s.calls == []

    # SpecMismatch propagates, also writing nothing.
    _edit(spec, "column: QTY_SHIPPED", "column: QTY")
    with pytest.raises(SpecMismatch):
        apply_to_tenant(s, spec, [csv_path], draft=True)
    assert s.calls == []


def test_apply_to_tenant_draft(project: Path) -> None:
    s = RecordingSession()
    loaded = load_project_spec(project / "specs" / "legacy_report.yaml")
    _, report = apply_to_tenant(s, loaded, [project.parent / "legacy_report.csv"], draft=True)
    assert report is not None and report.facts == 96
    assert s.jobs == [(f"wrangle-draft:legacy_report@{loaded.bundle_sha12}", "", "")]
    up = _upserts(s)
    assert set(up) == {
        ("plant", c)
        for c in (
            "Detroit Stamping 07",
            "Toledo Assembly 12",
            "Saltillo Castings 21",
            "Puebla Machining 33",
            "Monterrey Paint 45",
        )
    }


def test_ingest_replace_directly_with_adapter(results: dict[str, ApplyResult]) -> None:
    s = FakeSession()
    adapter = WrangleAdapter.from_result(results["erp_shipments"])
    report = ingest(s, adapter, on_new_version="replace")
    assert report.facts == 59 and report.entities == 5
    assert s.jobs[0][0].startswith("wrangle-draft:erp_shipments@")
    # Facts arrive with their FactRow intact apart from doc_id.
    raw = results["erp_shipments"].facts[0].row
    assert dataclasses.replace(s.facts_added[0], doc_id=None) == raw


# --- review fixes: spec changes reload, native cells, exact values, case, XLSX sheet --------


def test_document_text_changes_with_the_bundle(project: Path) -> None:
    """The same bytes under a corrected spec render a different document, so the ingest makes
    a new version and `replace` retires the facts of the wrong one."""
    spec = project / "specs" / "legacy_report.yaml"
    csv_path = project.parent / "legacy_report.csv"
    _edit(spec, "    scale: 1000\n    precision: 1000\n", "    scale: 1\n")
    wrong = run(spec, [csv_path])
    _edit(spec, "    scale: 1\n", "    scale: 1000\n    precision: 1000\n")
    fixed = run(spec, [csv_path])
    assert wrong.loaded.bundle_sha256 != fixed.loaded.bundle_sha256
    assert wrong.documents[0].source == fixed.documents[0].source
    assert wrong.documents[0].text != fixed.documents[0].text

    s = RecordingSession()
    _, first = apply_to_tenant(s, wrong.loaded, [csv_path], draft=True)
    assert first is not None and first.facts == 96
    _, second = apply_to_tenant(s, fixed.loaded, [csv_path], draft=True)
    assert second is not None
    assert (second.documents_changed, second.facts, second.facts_replaced) == (1, 96, 96)
    toledo = {
        r.value
        for r in s.facts_added
        if r.source_ref.startswith("legacy_report!C34@") and r.field == "shipped_units"
    }
    assert toledo == {39900}


def test_two_specs_over_one_file_are_two_documents(project: Path) -> None:
    other = project / "specs" / "erp_returns.yaml"
    text = (project / "specs" / "erp_shipments.yaml").read_text(encoding="utf-8")
    other.write_text(
        text.replace("spec: erp_shipments", "spec: erp_returns").replace(
            "field: shipped_units", "field: returned_units"
        ),
        encoding="utf-8",
    )
    csv_path = project.parent / "erp_shipments.csv"
    a = run(project / "specs" / "erp_shipments.yaml", [csv_path])
    b = run(other, [csv_path])
    assert a.documents[0].source == "erp_shipments/erp_shipments.csv#erp_shipments"
    assert b.documents[0].source == "erp_returns/erp_shipments.csv#erp_shipments"

    s = RecordingSession()
    apply_to_tenant(s, a.loaded, [csv_path], draft=True)
    _, rep = apply_to_tenant(s, b.loaded, [csv_path], draft=True)
    assert rep is not None and rep.facts == 59 and rep.facts_replaced == 0


def _mini_project(tmp_path: Path, spec_yaml: str) -> Path:
    proj = tmp_path / "mini"
    (proj / "specs").mkdir(parents=True)
    shutil.copy(PROJECT / "dictionary.yaml", proj / "dictionary.yaml")
    spec = proj / "specs" / "mini.yaml"
    spec.write_text(spec_yaml, encoding="utf-8")
    return spec


def test_native_xlsx_cells_skip_the_text_grammar(tmp_path: Path) -> None:
    """A European-style spec (`thousands: "."`) over an XLSX whose amounts are native numbers:
    the native 1234.5 is 1234.5, not a reject (or 1234 for 1.234). Text cells still use the
    declared grammar. Native date cells feed `date` and `period` without a strptime format."""
    from datetime import datetime

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S"
    ws.append(["Item", "Month", "Quarter", "Amount"])
    ws.append(["A", datetime(2026, 1, 15), datetime(2025, 11, 3), 1.234])
    ws.append(["B", datetime(2026, 2, 28, 13, 30), datetime(2026, 2, 1), 1234.5])
    ws.append(["C", "15/03/2026", datetime(2026, 3, 9), "1.234,5"])
    ws.append(["D", datetime(2026, 4, 1), datetime(2026, 4, 1), 12.5])
    xlsx = tmp_path / "eu.xlsx"
    wb.save(xlsx)
    spec = _mini_project(
        tmp_path,
        """spec: mini
dictionary: ../dictionary.yaml
source: {sheet: S, header_rows: [1]}
entity:
  kind: item
  parts: [{column: Item, ops: [{op: text}], resolve: {policy: open}}]
period:
  column: {name: Month, ops: [{op: date, format: "%d/%m/%Y"}], grain: month}
measures:
  - {field: line_total_usd, column: Amount, ops: [{op: number, thousands: ".", decimal: ","}]}
""",
    )
    result = run(spec, [xlsx])
    assert result.rejects == []
    got = {f.row.entity: (f.row.value, f.row.period_start) for f in result.facts}
    assert got == {
        "A": (Decimal("1.234"), date(2026, 1, 1)),
        "B": (Decimal("1234.5"), date(2026, 2, 1)),
        "C": (Decimal("1234.5"), date(2026, 3, 1)),  # text cell: the declared grammar
        "D": (Decimal("12.5"), date(2026, 4, 1)),
    }

    # A first `period` op on a date cell: the op's grain and fiscal year apply to the date.
    _edit(
        spec,
        '  column: {name: Month, ops: [{op: date, format: "%d/%m/%Y"}], grain: month}',
        '  column: {name: Quarter, ops: [{op: period, pattern: "Q{q} FY{yy}", grain: quarter,'
        " fy_start_month: 10}]}",
    )
    quarters = {
        f.row.entity: (f.row.period_start, f.row.period_end) for f in run(spec, [xlsx]).facts
    }
    assert quarters == {
        "A": (date(2025, 10, 1), date(2025, 12, 31)),
        "B": (date(2026, 1, 1), date(2026, 3, 31)),
        "C": (date(2026, 1, 1), date(2026, 3, 31)),
        "D": (date(2026, 4, 1), date(2026, 6, 30)),
    }

    # The read-back rule is the same function.
    grid = read_grid(xlsx, "S")
    fn = cell_pipeline([{"op": "number", "thousands": ".", "decimal": ","}])
    assert fn(grid.cell(3, 4)) == Decimal("1234.5")
    assert fn(grid.cell(4, 4)) == Decimal("1234.5")


def test_values_are_exact_and_outputs_serialise(tmp_path: Path, project: Path) -> None:
    path = project.parent / "variant" / "erp_exact.csv"
    path.parent.mkdir()
    path.write_text(
        "PLANT_CD,SHIP_DT,QTY_SHIPPED\n1007,20251031,1100.0000000000002\n"
        "1012,20251031,0.30000000000000004\n1021,20251031,1.00000000000000000001\n",
        encoding="utf-8",
    )
    result = run(project / "specs" / "erp_shipments.yaml", [path])
    values = [f.row.value for f in result.facts]
    assert values == [
        Decimal("1100.0000000000002"),
        Decimal("0.30000000000000004"),
        Decimal("1.00000000000000000001"),
    ]
    paths = write_outputs(result, tmp_path / "out")
    preview = [d["value"] for d in _jsonl(paths["preview"])]
    # A float that reads back exactly stays a number; one that can't is a string.
    assert preview == [1100.0000000000002, 0.30000000000000004, "1.00000000000000000001"]
    assert Decimal(repr(preview[0])) == values[0]


def test_open_values_differing_only_in_case_are_one_entity(tmp_path: Path) -> None:
    csv_path = tmp_path / "items.csv"
    csv_path.write_text(
        "Item,Month,Total\nWidget-a,2026-01,100\nWIDGET-A,2026-01,7\n"
        "widget-A,2026-02,5\ndetroit stamping 07,2026-01,1\n",
        encoding="utf-8",
    )
    spec = _mini_project(
        tmp_path,
        """spec: mini
dictionary: ../dictionary.yaml
crosswalks:
  plants: {path: ../crosswalks/plants.csv, kind: plant}
source: {header_rows: [1]}
entity:
  kind: item
  parts: [{column: Item, ops: [{op: text}], resolve: {policy: open}}]
period: {column: {name: Month, ops: [{op: date, format: "%Y-%m"}], grain: month}}
measures: [{field: line_total_usd, column: Total, ops: [{op: number}]}]
""",
    )
    (spec.parent.parent / "crosswalks").mkdir()
    shutil.copy(PROJECT / "crosswalks" / "plants.csv", spec.parent.parent / "crosswalks")
    result = run(spec, [csv_path])
    assert [(f.row.entity, f.row.value) for f in result.facts] == [
        ("Widget-a", 100),
        ("Widget-a", 7),
        ("Widget-a", 5),
        ("Detroit Stamping 07", 1),  # a crosswalk canonical's spelling
    ]
    assert result.report["entities"] == {"exact": 0, "normalised": 3, "fuzzy": 0, "new": 1}
    assert [(e.kind, e.canonical_name) for e in result.files[0].entities] == [("item", "Widget-a")]
    # Now one key with two values, so the within-load conflict is reported.
    assert result.report["conflicts_within_load"] == [
        {
            "document": "mini/items.csv#items",
            "key": ["Widget-a", "line_total_usd", "2026-01"],
            "cells": ["items!C2", "items!C3"],
        }
    ]


def test_composite_names_merge_casefolded(tmp_path: Path) -> None:
    """An open part spelled two ways at one site: one composite entity, first spelling."""
    csv_path = tmp_path / "parts.csv"
    csv_path.write_text(
        "Site,Part,Month,Total\nDET-07,p-1,2026-01,10\nDetroit Plt 07,P-1,2026-02,20\n"
        "TOL-12,P-1,2026-01,30\n",
        encoding="utf-8",
    )
    spec = _mini_project(
        tmp_path,
        """spec: mini
dictionary: ../dictionary.yaml
crosswalks:
  plants: {path: ../crosswalks/plants.csv, kind: plant}
source: {header_rows: [1]}
entity:
  kind: plant_part
  parts:
    - {column: Site, resolve: {crosswalk: plants}}
    - {column: Part, resolve: {policy: open}}
  template: "{0}/{1}"
period: {column: {name: Month, ops: [{op: date, format: "%Y-%m"}], grain: month}}
measures: [{field: line_total_usd, column: Total, ops: [{op: number}]}]
""",
    )
    (spec.parent.parent / "crosswalks").mkdir()
    shutil.copy(PROJECT / "crosswalks" / "plants.csv", spec.parent.parent / "crosswalks")
    result = run(spec, [csv_path])
    assert [f.row.entity for f in result.facts] == [
        "Detroit Stamping 07/p-1",
        "Detroit Stamping 07/p-1",
        "Toledo Assembly 12/p-1",
    ]
    assert [e.canonical_name for e in result.files[0].entities] == [
        "Detroit Stamping 07/p-1",
        "Toledo Assembly 12/p-1",
    ]


def test_xlsx_without_source_sheet_is_spec_mismatch(project: Path) -> None:
    spec = project / "specs" / "planner_forecast.yaml"
    _edit(spec, "  sheet: Plan\n", "")
    with pytest.raises(SpecMismatch, match="XLSX needs source.sheet"):
        run(spec, [project.parent / "planner_forecast.xlsx"])


def test_native_rule_applies_after_leading_text_ops() -> None:
    from fde_kit.wrangle.apply import cell_pipeline
    from fde_kit.wrangle.read import Cell

    fn = cell_pipeline([{"op": "text"}, {"op": "number", "thousands": ".", "decimal": ","}])
    assert fn(Cell(row=2, col=3, raw="1234.5", kind="number")) == Decimal("1234.5")
    assert fn(Cell(row=2, col=3, raw=" 1.234,5 ", kind="text")) == Decimal("1234.5")


# --- nulls, sections, multi-column periods --------------------------------------------------

_NULLS_SPEC = """spec: mini
dictionary: ../dictionary.yaml
source: {header_rows: [1]}
entity:
  kind: item
  parts: [{column: Item, resolve: {policy: open}}]
period:
  unpivot:
    columns: {from: "2026-01", to: "2026-02"}
    ops: [{op: period, format: "%Y-%m", grain: month}]
measures:
  - {field: line_total_usd, ops: [{op: number}], nulls: ["--", "(S)"]}
"""


def test_nulls_are_empty_cells_counted_by_placeholder(tmp_path: Path) -> None:
    csv_path = tmp_path / "n.csv"
    csv_path.write_text(
        'Item,2026-01,2026-02\nA,1,--\nB, (S) ,2\nC,(S),\nD,(D),3\n"(S)",4,5\n',
        encoding="utf-8",
    )
    spec = _mini_project(tmp_path, _NULLS_SPEC)
    result = run(spec, [csv_path])
    got = sorted((f.row.entity, f.row.period_start.month, f.row.value) for f in result.facts)
    # `(D)` isn't listed, so it's a reject, not a silent null; nulls don't apply to entities.
    assert got == [("(S)", 1, 4), ("(S)", 2, 5), ("A", 1, 1), ("B", 2, 2), ("D", 2, 3)]
    f = result.report["files"][0]
    assert f["null_skipped"] == {"(S)": 2, "--": 1} and f["empty_skipped"] == 1
    assert list(f["null_skipped"]) == ["(S)", "--"]  # sorted, so the report is stable
    assert [(r["cell"], r["raw"], r["reason"]) for r in result.rejects] == [
        ("B5", "(D)", "number: parentheses not allowed")
    ]

    # `empty: reject` rejects a placeholder as it would an empty cell, naming the placeholder.
    _edit(spec, 'nulls: ["--", "(S)"]}', 'nulls: ["--", "(S)"], empty: reject}')
    strict = run(spec, [csv_path])
    nulls = [r for r in strict.rejects if r["op"] == "nulls"]
    assert [(r["cell"], r["raw"], r["reason"], r["stage"]) for r in nulls] == [
        ("C2", "--", "null placeholder", "measure"),
        ("B3", " (S) ", "null placeholder", "measure"),
        ("B4", "(S)", "null placeholder", "measure"),
    ]
    assert strict.report["files"][0]["null_skipped"] == {}

    # Without `nulls` the same cells are number rejects: no placeholder is implied.
    _edit(spec, ', nulls: ["--", "(S)"], empty: reject}', "}")
    plain = run(spec, [csv_path])
    assert plain.report["rejects"]["by_reason"] == {
        "number: parentheses not allowed": 3,
        "number: unparseable": 1,
    }
    assert plain.report["files"][0]["null_skipped"] == {}


_TWO_TABLES = (
    "Kind,2026-01\n"
    "NOT ADJUSTED,\n"
    "Total,100\n"
    "Food,40\n"
    "ADJUSTED(2),\n"
    "Total,110\n"
    "Food,44\n"
    "(2) Adjusted for seasonal variation,\n"
)
_SECTION_SPEC = """spec: mini
dictionary: ../dictionary.yaml
source:
  header_rows: [1]
  {bounds}
  skip_rows: [{{blank: ["2026-01"]}}]
entity:
  kind: kind
  parts: [{{column: Kind, resolve: {{policy: open}}}}]
period:
  unpivot:
    columns: {{from: "2026-01", to: "2026-01"}}
    ops: [{{op: period, format: "%Y-%m", grain: month}}]
measures:
  - {{field: line_total_usd, ops: [{{op: number}}]}}
"""


def _section_run(tmp_path: Path, name: str, bounds: str) -> ApplyResult:
    csv_path = tmp_path / "two.csv"
    csv_path.write_text(_TWO_TABLES, encoding="utf-8")
    spec = _mini_project(tmp_path / name, _SECTION_SPEC.format(bounds=bounds))
    return run(spec, [csv_path])


def _cells(result: ApplyResult) -> list[tuple[str, object, str]]:
    return [(r.fact.row.entity, r.fact.row.value, r.cell) for f in result.files for r in f.facts]


def test_start_after_and_stop_before_pick_one_table(tmp_path: Path) -> None:
    both = _section_run(tmp_path, "nsa", 'stop_before: {column: Kind, matches: "^ADJUSTED"}')
    assert _cells(both) == [
        ("Total", 100, "two!B3"),
        ("Food", 40, "two!B4"),
    ]
    f = both.report["files"][0]
    assert f["section"] == {"stop_before_row": 5}
    assert f["rows_read"] == 3 and f["rows_skipped"] == {"skip_rows[0]": 1}  # NOT ADJUSTED

    sa = _section_run(tmp_path, "sa", 'start_after: {column: Kind, matches: "^ADJUSTED"}')
    assert _cells(sa) == [
        ("Total", 110, "two!B6"),
        ("Food", 44, "two!B7"),
    ]
    assert sa.report["files"][0]["section"] == {"start_after_row": 5}
    assert sa.report["conflicts_within_load"] == []

    # Both bounds: the first stop_before *after* the start counts, not one above it.
    nsa = _section_run(
        tmp_path,
        "nsa2",
        'start_after: {column: Kind, matches: "^NOT ADJUSTED$"}\n'
        '  stop_before: {column: Kind, matches: "^(ADJUSTED|Total)"}',
    )
    assert _cells(nsa) == []  # Total (row 3) stops it at once
    assert nsa.report["files"][0]["section"] == {"start_after_row": 2, "stop_before_row": 3}

    # Without bounds the two tables load into the same keys: a conflict, which is the problem.
    neither = _section_run(tmp_path, "neither", "")
    assert len(neither.report["conflicts_within_load"]) == 2
    assert "section" not in neither.report["files"][0]


def test_section_bound_that_matches_nothing_is_spec_mismatch(tmp_path: Path) -> None:
    with pytest.raises(SpecMismatch, match="source.start_after: no row"):
        _section_run(tmp_path, "a", 'start_after: {column: Kind, matches: "^SEASONAL"}')
    with pytest.raises(SpecMismatch, match="source.stop_before: no row"):
        _section_run(tmp_path, "b", 'stop_before: {column: Kind, matches: "^Kind$"}')  # header
    with pytest.raises(SpecMismatch, match="source.start_after column 'Nope'"):
        _section_run(tmp_path, "c", 'start_after: {column: Nope, matches: "x"}')


def test_period_from_two_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "gen.csv"
    csv_path.write_text(
        "YEAR,MONTH,STATE,MWH\n2024,1,CT,10\n2024, 12 ,CT,20\n2024,13,CT,30\n2024,,CT,40\n"
        ",,CT,50\n",
        encoding="utf-8",
    )
    spec = _mini_project(
        tmp_path,
        """spec: mini
dictionary: ../dictionary.yaml
source: {header_rows: [1]}
entity:
  kind: state
  parts: [{column: STATE, resolve: {policy: open}}]
period:
  column:
    name: [YEAR, MONTH]
    ops: [{op: period, format: "%Y-%m", grain: month}]
measures: [{field: line_total_usd, column: MWH, ops: [{op: number}]}]
""",
    )
    result = run(spec, [csv_path])
    recs = [r for f in result.files for r in f.facts]
    assert [(r.fact.row.value, r.fact.row.period_start, r.fact.row.period_end) for r in recs] == [
        (10, date(2024, 1, 1), date(2024, 1, 31)),
        (20, date(2024, 12, 1), date(2024, 12, 31)),  # each value is whitespace-cleaned
    ]
    assert [(r.period_from, r.period_cells) for r in recs] == [
        ("A2+B2", ("A2", "B2")),
        ("A3+B3", ("A3", "B3")),
    ]
    # The joined value is what the ops saw, so a reject shows it.
    assert [(r["row"], r["raw"], r["stage"], r["reason"]) for r in result.rejects] == [
        (4, "2024-13", "period", "period: unparseable"),
        (5, "2024-", "period", "period: unparseable"),
        (6, "", "period", "period: empty"),
    ]
    paths = write_outputs(result, tmp_path / "out")
    preview = _jsonl(paths["preview"])
    assert preview[0]["period_from"] == "A2+B2" and preview[0]["period_cells"] == ["A2", "B2"]

    # A single name keeps the old trail: one cell.
    _edit(spec, "    name: [YEAR, MONTH]\n", "    name: YEAR\n")
    _edit(spec, 'format: "%Y-%m", grain: month', 'format: "%Y", grain: year')
    one = [r for f in run(spec, [csv_path]).files for r in f.facts]
    assert one[0].period_from == "A2" and one[0].period_cells == ("A2",)
