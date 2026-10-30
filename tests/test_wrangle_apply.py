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
from pathlib import Path

import pytest

from fde_kit.rag.adapters.base import AdapterBatch, ingest
from fde_kit.wrangle.adapter import WrangleAdapter, apply_to_tenant, job_kind_for
from fde_kit.wrangle.apply import (
    ApplyResult,
    SpecMismatch,
    gate,
    parse_source_ref,
    run,
    write_outputs,
)
from fde_kit.wrangle.labels import header_fingerprint, header_labels
from fde_kit.wrangle.parse import compile_ops, to_fact_value
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
    pipelines = {m.field: compile_ops(m.ops) for m in loaded.spec.measures if m.ops}
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
            expected = str(pipelines[m.field](cell.raw)) if m.ops else cell.raw
        else:
            expected = to_fact_value(pipelines[m.field](cell.raw) * m.scale_decimal)
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
    assert doc.source == "planner-forecast#Plan"
    assert doc.title == "planner_forecast.xlsx — Plan"
    rows = list(csv.reader(io.StringIO(doc.text)))
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
    path = FIXTURES / "erp_shipments.csv"
    assert edoc.source == f"{path}#erp_shipments"  # default key: the path as passed
    assert edoc.title == "erp_shipments.csv — erp_shipments"
    erows = list(csv.reader(io.StringIO(edoc.text)))
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


def test_two_files_in_one_load_share_duplicate_detection(project: Path) -> None:
    a = FIXTURES / "erp_shipments.csv"
    b = _erp_variant(project, "", name="erp_copy.csv")
    result = run(project / "specs" / "erp_shipments.yaml", [a, b])
    # The copy is a different sheet (file stem) with identical values: every fact a duplicate.
    assert result.report["facts"] == 59
    assert result.report["duplicates_dropped"] == 1 + 60
    assert len(result.files) == 2 and result.files[1].facts == []
    assert len(result.rejects) == 2


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
    assert [len(b.facts) for b in batches] == [59, 0]


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

    title, text, meta = s.documents["planner-forecast#Plan"]
    assert title == "planner_forecast.xlsx — Plan" and text.startswith("row,A,B,")
    assert meta["spec"] == "planner_forecast" and meta["bundle_sha256"] == loaded.bundle_sha256
    assert len(s.facts_added) == 35
    assert {r.doc_id for r in s.facts_added} == {"doc:planner-forecast#Plan"}
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
    assert ("delete_facts_for_document", "doc:planner-forecast#Plan") in s.calls
    assert len(s.facts_added) == 35
    assert {r.doc_id for r in s.facts_added} == {"doc:planner-forecast#Plan@v2"}
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
