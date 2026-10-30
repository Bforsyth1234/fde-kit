"""`fde wrangle` driven through `cli.main`: profile, check, dry-run, approve, and apply's gates.

No Postgres here: apply is tested up to the point where it would open the store (argument
errors, and refusals that must happen before any connection), plus its success path against a
stubbed `apply_to_tenant`. The real load runs in CI (`fde wrangle apply` into `wrangle-ci`).
"""

import io
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from fde_kit import cli

FIXTURES = Path(__file__).parent / "fixtures" / "wrangle"
SPECS = FIXTURES / "project" / "specs"


@pytest.fixture
def out(monkeypatch):
    """Capture `cli.console` as plain text at a wide width; returns a getter."""
    buf = io.StringIO()
    monkeypatch.setattr(
        cli, "console", Console(file=buf, width=200, force_terminal=False, color_system=None)
    )
    return buf.getvalue


@pytest.fixture
def project(tmp_path) -> Path:
    """A writable copy of the fixture project (approve and crosswalk edits never touch the
    committed files)."""
    dst = tmp_path / "project"
    shutil.copytree(FIXTURES / "project", dst)
    return dst


def _line_with(text: str, needle: str) -> str:
    return next(line for line in text.splitlines() if needle in line)


# --- profile


def test_profile_human(out):
    assert cli.main(["wrangle", "profile", str(FIXTURES / "planner_forecast.xlsx")]) == 0
    text = out()
    assert "planner_forecast.xlsx — Plan" in text
    assert "header rows [4, 5]" in text
    assert "Forecast / Nov-26" in text and "x1000" in text
    assert "unpivot candidate Forecast / Nov-26 .. Forecast / Feb-27" in text
    assert "scale hint x1000 from A2: Units (000s)" in text
    assert "skip candidate row 20: label+sum 'Total'" in text
    assert "key candidate: Site + Part No" in text


def test_profile_json(capsys):
    path = FIXTURES / "planner_forecast.xlsx"
    assert cli.main(["wrangle", "profile", str(path), "--sheet", "Plan", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sheet"] == "Plan" and data["header_rows"] == [4, 5]
    assert data["unpivot_candidates"][0]["from"] == "Forecast / Nov-26"
    assert len(data["profile_sha256"]) == 64


def test_profile_bad_args(out):
    assert cli.main(["wrangle", "profile", str(FIXTURES / "nope.csv")]) == 2
    assert "not a file" in out()
    path = str(FIXTURES / "planner_forecast.xlsx")
    assert cli.main(["wrangle", "profile", path, "--sheet", "NoSuchSheet"]) == 1


# --- check / approve


def test_check_valid_spec(out):
    assert cli.main(["wrangle", "check", str(SPECS / "erp_shipments.yaml")]) == 0
    text = out()
    sha = _line_with(text, "bundle_sha256").split()[-1]
    assert len(sha) == 64
    assert "not approved" in text and "ok:" in text


def test_check_colliding_crosswalk(out, project):
    cw = project / "crosswalks" / "plants.csv"
    # `det_07` normalises to `det7`, as `DET-07` (Detroit) does, but maps to Toledo.
    cw.write_text(cw.read_text() + "det_07,Toledo Assembly 12\n")
    assert cli.main(["wrangle", "check", str(project / "specs" / "erp_shipments.yaml")]) == 1
    text = out()
    assert "FAIL" in text and "'DET-07', 'det_07' all normalise to 'det7'" in text
    assert "ok:" not in text


def test_check_spec_error(out, project):
    spec = project / "specs" / "erp_shipments.yaml"
    spec.write_text(spec.read_text().replace("dictionary: ../dictionary.yaml", "dictionary: x.y"))
    assert cli.main(["wrangle", "check", str(spec)]) == 1
    assert "FAIL" in out() and "cannot read" in out()


def test_approve_then_check_then_stale(out, project):
    spec = str(project / "specs" / "erp_shipments.yaml")
    assert cli.main(["wrangle", "approve", spec, "--by", "brooks"]) == 0
    assert "approved erp_shipments at bundle" in out()

    assert cli.main(["wrangle", "check", spec]) == 0
    assert "approved by brooks" in _line_with(out(), "approved by")

    cw = project / "crosswalks" / "plants.csv"
    data = cw.read_bytes()
    cw.write_bytes(data.replace(b"Detroit Stamping 07\n", b"Detroit Stamping 07 \n", 1))
    assert cli.main(["wrangle", "check", spec]) == 0  # still valid; the approval is stale
    stale = _line_with(out(), "stale approval")
    assert "by brooks" in stale and "the bundle is now" in stale


def test_approve_refused_when_check_fails(out, project):
    cw = project / "crosswalks" / "plants.csv"
    cw.write_text(cw.read_text() + "det_07,Toledo Assembly 12\n")
    spec = project / "specs" / "erp_shipments.yaml"
    before = spec.read_bytes()
    assert cli.main(["wrangle", "approve", str(spec), "--by", "brooks"]) == 1
    assert "refused" in out()
    assert spec.read_bytes() == before


# --- dry-run


def test_dry_run_summary_and_outputs(out, tmp_path):
    dest = tmp_path / "out"
    argv = [
        "wrangle",
        "dry-run",
        str(SPECS / "erp_shipments.yaml"),
        str(FIXTURES / "erp_shipments.csv"),
        "--out",
        str(dest),
    ]
    assert cli.main(argv) == 0  # rejects don't fail a dry run
    text = out()
    assert "facts 59  rejects 1 (rate 0.0167)  duplicates dropped 1" in text
    assert "reject 1 x number: unparseable" in text
    assert "entities by strategy: exact 5  normalised 0  fuzzy 0  new 0" in text
    assert "conflicts within load: 0" in text
    assert "apply now would refuse" in text and "is not approved" in text

    for name in ("report.json", "rejects.jsonl", "facts.preview.jsonl", "crosswalk.proposed.csv"):
        assert (dest / name).is_file(), name
    report = json.loads((dest / "report.json").read_text())
    assert report["facts"] == 59 and report["duplicates_dropped"] == 1
    rejects = (dest / "rejects.jsonl").read_text().splitlines()
    assert len(rejects) == 1 and json.loads(rejects[0])["cell"] == "C28"
    assert len((dest / "facts.preview.jsonl").read_text().splitlines()) == 50


def test_dry_run_approved_spec_would_load(out, project):
    spec = str(project / "specs" / "planner_forecast.yaml")
    assert cli.main(["wrangle", "approve", spec, "--by", "ci"]) == 0
    assert cli.main(["wrangle", "dry-run", spec, str(FIXTURES / "planner_forecast.xlsx")]) == 0
    text = out()
    assert "facts 35  rejects 0" in text and "normalised 1" in text
    assert "apply now would load" in text


def test_dry_run_spec_mismatch_is_exit_1(out):
    # The ERP spec against the legacy file: its columns aren't there.
    argv = [
        "wrangle",
        "dry-run",
        str(SPECS / "erp_shipments.yaml"),
        str(FIXTURES / "legacy_report.csv"),
    ]
    assert cli.main(argv) == 1
    assert "spec mismatch" in out()


def test_dry_run_exact_decimal_values_and_same_key_files(out, tmp_path):
    """Non-integral values are Decimals: the summary and the JSON outputs handle them. Two
    files with one rendered document_key are a spec mismatch (exit 1)."""
    data = tmp_path / "erp_exact.csv"
    data.write_text(
        "PLANT_CD,SHIP_DT,QTY_SHIPPED\n1007,20251031,1100.0000000000002\n", encoding="utf-8"
    )
    dest = tmp_path / "out"
    spec = str(SPECS / "erp_shipments.yaml")
    assert cli.main(["wrangle", "dry-run", spec, str(data), "--out", str(dest)]) == 0
    assert "facts 1  rejects 0" in out()
    preview = json.loads((dest / "facts.preview.jsonl").read_text())
    assert preview["value"] == 1100.0000000000002

    assert cli.main(["wrangle", "dry-run", spec, str(data), str(data)]) == 1
    assert "spec mismatch" in out() and "supersede each other" in out()


def test_dry_run_missing_file_is_exit_2(out):
    argv = ["wrangle", "dry-run", str(SPECS / "erp_shipments.yaml"), "nope.csv"]
    assert cli.main(argv) == 2


# --- apply (no database)


@pytest.fixture
def no_store(monkeypatch):
    """Fail the test if apply tries to open the store."""

    def boom(_args):
        raise AssertionError("apply opened the store")

    monkeypatch.setattr(cli, "_adapter_store", boom)


def test_apply_needs_tenant(capsys):
    argv = [
        "wrangle",
        "apply",
        str(SPECS / "erp_shipments.yaml"),
        str(FIXTURES / "erp_shipments.csv"),
    ]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "--tenant" in capsys.readouterr().err


def test_apply_unapproved_refused_before_store(out, no_store, tmp_path):
    dest = tmp_path / "out"
    argv = [
        "wrangle",
        "apply",
        str(SPECS / "erp_shipments.yaml"),
        str(FIXTURES / "erp_shipments.csv"),
        "--tenant",
        "t1",
        "--out",
        str(dest),
    ]
    assert cli.main(argv) == 1
    text = out()
    assert "refused, nothing written" in text and "is not approved" in text
    assert (dest / "report.json").is_file() and (dest / "rejects.jsonl").is_file()


def test_apply_reject_rate_refused_before_store(out, no_store, project):
    spec = str(project / "specs" / "erp_shipments.yaml")
    assert cli.main(["wrangle", "approve", spec, "--by", "ci"]) == 0
    argv = [
        "wrangle",
        "apply",
        spec,
        str(FIXTURES / "erp_shipments.csv"),
        "--tenant",
        "t1",
        "--max-reject-rate",
        "0.01",
    ]
    assert cli.main(argv) == 1
    text = out()
    assert "reject rate 0.0167 (1 rejects) is over the limit 0.01" in text
    assert "is not approved" not in text


def test_apply_loads_when_gates_pass(out, monkeypatch, project):
    """The success path's wiring, with the store and the load stubbed out."""
    from fde_kit.rag.adapters.base import IngestReport
    from fde_kit.wrangle import adapter, run

    calls = []

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @contextmanager
        def session(self, tenant):
            calls.append(("session", tenant))
            yield "S"

    def fake_apply(session, loaded, files, *, draft, max_reject_rate, force):
        calls.append(("apply", session, draft, max_reject_rate, force))
        report = IngestReport(1, 1, 35, 0, 8, "abcdef0123456789", facts_replaced=3)
        return run(loaded, files, force=force), report

    monkeypatch.setattr(cli, "_adapter_store", lambda args: Store())
    monkeypatch.setattr(adapter, "apply_to_tenant", fake_apply)
    argv = [
        "wrangle",
        "apply",
        str(project / "specs" / "planner_forecast.yaml"),
        str(FIXTURES / "planner_forecast.xlsx"),
        "--tenant",
        "acme",
        "--draft",
    ]
    assert cli.main(argv) == 0
    assert calls == [("session", "acme"), ("apply", "S", True, 0.02, False)]
    text = out()
    row = _line_with(text, "abcdef01")
    assert [c.strip() for c in row.split("│")[1:7]] == ["8", "1", "1", "35", "3", "0"]
    assert "job kind wrangle-draft:planner_forecast@" in text


def test_profile_csv_encoding_guessed_or_given(out, tmp_path):
    p = tmp_path / "counties.csv"
    p.write_bytes("STNAME,CTYNAME,POP\nNew Mexico,Doña Ana County,1\nX,Y,2\n".encode("latin-1"))
    assert cli.main(["wrangle", "profile", str(p)]) == 0
    text = out()
    assert "encoding cp1252" in text
    assert "warning" in text and "decoded as cp1252" in text


def test_profile_csv_encoding_flag(capsys, tmp_path):
    p = tmp_path / "counties.csv"
    p.write_bytes("STNAME,CTYNAME,POP\nNew Mexico,Doña Ana County,1\nX,Y,2\n".encode("latin-1"))
    assert cli.main(["wrangle", "profile", str(p), "--encoding", "latin-1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["encoding"] == "latin-1" and data["warnings"] == []


def test_profile_csv_wrong_encoding_flag(out, tmp_path):
    p = tmp_path / "counties.csv"
    p.write_bytes("A,B\nDoña,1\n".encode("latin-1"))
    assert cli.main(["wrangle", "profile", str(p), "--encoding", "utf-8"]) == 1
    assert "byte offset" in out() and "source.encoding" in out()
