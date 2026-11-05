"""The wrangle benchmark (docs/wrangle-plan.md §5): the world, the scorers, the baseline, the gate.

The scorers are what later PRs get judged by, so each one is shown to move: the correct project
scores 100%, and a planted error of each kind lowers exactly the metric that should catch it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("openpyxl")
pytest.importorskip("rapidfuzz")
yaml = pytest.importorskip("yaml")

from evals.wrangle.bench import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_SEED,
    bench,
    check_gate,
    improvements,
    main,
    run_mapper,
    self_check,
)
from evals.wrangle.generator import SOURCE_FILES, generate  # noqa: E402
from evals.wrangle.mapper import (  # noqa: E402
    HeuristicMapper,
    TruthMapper,
    match_account,
    match_plant,
)
from evals.wrangle.scorers import load_truth, score_all, score_rejects  # noqa: E402
from fde_kit import cli  # noqa: E402
from fde_kit.wrangle.read import read_grid  # noqa: E402


class EditedTruth(TruthMapper):
    """The correct project with one planted error."""

    def __init__(self, edit):
        self.edit = edit

    def propose(self, world: Path, project_dir: Path):
        proposals = super().propose(world, project_dir)
        self.edit(project_dir)
        return proposals


def _score(tmp_path: Path, mapper) -> dict:
    return bench(DEFAULT_SEED, mapper, tmp_path)


def _edit_spec(project: Path, name: str, fn) -> None:
    path = project / "specs" / f"{name}.yaml"
    spec = yaml.safe_load(path.read_text())
    fn(spec)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


@pytest.mark.parametrize("seed", [7, 11, 12, 13])
def test_correct_project_scores_100_percent(tmp_path: Path, seed: int) -> None:
    result = bench(seed, TruthMapper(), tmp_path)
    assert self_check(result) == []
    assert set(result["by_mess"].values()) == {1.0}
    assert result["cost"]["llm_calls"] == 0


def test_world_matches_the_plan(tmp_path: Path) -> None:
    world = generate(DEFAULT_SEED, tmp_path)
    assert sorted(p.name for p in (world / "sources").iterdir()) == sorted(SOURCE_FILES.values())
    facts = [json.loads(x) for x in (world / "truth" / "facts.jsonl").read_text().splitlines()]
    plants = {f["entity"] for f in facts if f["source"] == "A"}
    parts = {f["entity"] for f in facts if f["source"] == "B"}
    months = {f["period_start"] for f in facts if f["source"] == "A"}
    assert (len(plants), len(parts), len(months)) == (20, 300, 24)
    assert (world / "truth" / "conflicts.jsonl").read_text()
    assert (world / "truth" / "rejects.jsonl").read_text()


def test_every_fact_mess_is_logged_and_every_type_touches_a_fact(tmp_path: Path) -> None:
    world = generate(DEFAULT_SEED, tmp_path)
    log = [json.loads(x) for x in (world / "truth" / "messes.jsonl").read_text().splitlines()]
    ids = {m["id"]: m["type"] for m in log}
    assert len(ids) == len(log), "mess ids are unique"
    used = set()
    for line in (world / "truth" / "facts.jsonl").read_text().splitlines():
        used |= set(json.loads(line)["messes"])
    assert used <= set(ids)
    # A closed plant's rows give no fact: it's scored by the rejects instead.
    assert {t for i, t in ids.items() if i not in used} <= {"closed_plant"}


def _world_bytes(world: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for p in sorted(world.rglob("*")):
        if p.suffix == ".xlsx":
            out[p.name] = [
                [c.raw for row in read_grid(p, sheet).rows for c in row] for sheet in _sheets(p)
            ]
        elif p.is_file():
            out[str(p.relative_to(world))] = p.read_bytes()
    return out


def _sheets(path: Path) -> list[str]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def test_same_seed_same_world_other_seed_other_world(tmp_path: Path) -> None:
    a = _world_bytes(generate(3, tmp_path / "a"))
    assert a == _world_bytes(generate(3, tmp_path / "b"))
    assert a != _world_bytes(generate(4, tmp_path / "c"))


def test_heuristic_mapper_does_not_read_the_answers(tmp_path: Path) -> None:
    """Delete `truth/`: the baseline must propose the same project without it."""
    world = generate(DEFAULT_SEED, tmp_path / "world")
    with_truth = _proposed_specs(world, tmp_path / "one")
    shutil.rmtree(world / "truth")
    assert _proposed_specs(world, tmp_path / "two") == with_truth


def _proposed_specs(world: Path, project: Path) -> dict[str, str]:
    HeuristicMapper().propose(world, project)
    return {p.name: p.read_text() for p in project.rglob("*") if p.is_file()}


# --- each planted error lowers the metric that should catch it --------------------------------


def test_swapped_plant_alias_is_a_false_merge_weighted_five_times(tmp_path: Path) -> None:
    def swap(project: Path) -> None:
        path = project / "crosswalks" / "plants.csv"
        lines = path.read_text().splitlines()
        alias, canonical = lines[1].split(",")
        other = next(x.split(",")[1] for x in lines[1:] if x.split(",")[1] != canonical)
        lines[1] = f"{alias},{other}"
        path.write_text("\n".join(lines) + "\n")

    m = _score(tmp_path, EditedTruth(swap))
    assert m["metrics"]["false_merges"] == 1
    n = m["metrics"]["entity_recall"]
    assert 0.9 < n < 1
    assert m["metrics"]["entity_precision"] < n, "false merges cost more than misses"
    assert m["false_merge_examples"]


def test_dropped_plant_alias_is_a_miss_not_a_false_merge(tmp_path: Path) -> None:
    def drop(project: Path) -> None:
        path = project / "crosswalks" / "plants.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], *lines[2:]]) + "\n")

    m = _score(tmp_path, EditedTruth(drop))["metrics"]
    assert m["false_merges"] == 0
    assert m["entity_misses"] == 1
    assert m["entity_precision"] == 1.0
    assert m["entity_recall"] < 1.0


def test_wrong_scale_lowers_column_mapping_and_value_accuracy(tmp_path: Path) -> None:
    def no_scale(spec: dict) -> None:
        spec["measures"][0]["scale"] = 1
        spec["measures"][0].pop("precision")

    m = _score(tmp_path, EditedTruth(lambda p: _edit_spec(p, "planner_forecast", no_scale)))
    assert m["metrics"]["column_mapping_precision"] < 1.0
    assert m["metrics"]["column_mapping_recall"] < 1.0
    assert m["by_source"]["B"] == 0
    assert m["by_mess"]["scale_000s"] == 0
    assert m["by_source"]["A"] == 1.0


def test_lost_subtotal_rule_shows_as_wrong_rejects(tmp_path: Path) -> None:
    def no_skip(spec: dict) -> None:
        spec["source"].pop("skip_rows")

    m = _score(tmp_path, EditedTruth(lambda p: _edit_spec(p, "planner_forecast", no_skip)))
    assert m["metrics"]["reject_precision"] < 1.0
    assert m["metrics"]["reject_recall"] == 1.0
    assert m["metrics"]["value_accuracy"] == 1.0, "every real fact still loaded"


def test_coa_alias_error_lowers_coa_accuracy(tmp_path: Path) -> None:
    def wrong(project: Path) -> None:
        path = project / "crosswalks" / "coa.csv"
        lines = path.read_text().splitlines()
        a, b = lines[1].split(","), lines[2].split(",")
        lines[1], lines[2] = f"{a[0]},{b[1]}", f"{b[0]},{a[1]}"
        path.write_text("\n".join(lines) + "\n")

    m = _score(tmp_path, EditedTruth(wrong))["metrics"]
    assert m["coa_accuracy"] < 1.0
    assert m["value_accuracy"] < 1.0


def test_one_spec_for_both_regions_misreads_european_numbers(tmp_path: Path) -> None:
    """The mess the baseline can't see: a European `(1.204)` read with `.` as the decimal is
    -1.204, a clean parse and a wrong number."""

    def one_format(spec: dict) -> None:
        spec["source"].pop("skip_rows")

    def edit(project: Path) -> None:
        _edit_spec(project, "legacy_report_na", one_format)
        (project / "specs" / "legacy_report_eu.yaml").unlink()

    class OneSpec(EditedTruth):
        def propose(self, world, project_dir):
            kept = [p for p in super().propose(world, project_dir) if p.spec_path.exists()]
            return kept

    m = _score(tmp_path, OneSpec(edit))
    assert m["by_mess"]["european_decimals"] < 1.0
    assert m["by_source"]["C"] < 1.0
    assert m["by_source"]["A"] == 1.0


def test_changed_source_cell_fails_the_provenance_read_back(tmp_path: Path) -> None:
    world = generate(DEFAULT_SEED, tmp_path / "world")
    project = tmp_path / "proposed"
    loads, _ = run_mapper(TruthMapper(), world, project)
    assert score_all(world, loads, project)["metrics"]["provenance_accuracy"] == 1.0
    path = world / "sources" / SOURCE_FILES["A"]
    lines = path.read_text().splitlines()
    cells = lines[1].split(",")
    cells[2] = str(int(cells[2].strip()) + 1)
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")
    assert score_all(world, loads, project)["metrics"]["provenance_accuracy"] < 1.0


def test_wrong_period_grain_is_not_a_right_fact(tmp_path: Path) -> None:
    """A forecast loaded as one-day periods has the right number on the wrong period."""

    def day_grain(spec: dict) -> None:
        spec["period"]["unpivot"]["ops"][-1]["grain"] = "day"

    m = _score(tmp_path, EditedTruth(lambda p: _edit_spec(p, "planner_forecast", day_grain)))
    assert m["by_source"]["B"] == 0
    assert m["by_source"]["A"] == 1.0
    assert m["metrics"]["phantom_facts"] > 0


def test_text_measure_is_read_back_as_text(tmp_path: Path) -> None:
    """A text measure loads as a string; the read-back must not try to scale it."""

    def add_text(spec: dict) -> None:
        spec["measures"].append(
            {"field": "onhand_units", "column": "Region", "ops": [{"op": "text", "strip": True}]}
        )

    m = _score(tmp_path, EditedTruth(lambda p: _edit_spec(p, "legacy_report_na", add_text)))
    assert m["metrics"]["provenance_accuracy"] == 1.0
    assert m["metrics"]["phantom_facts"] > 0


def test_two_measures_for_one_field_read_back_through_their_own_ops(tmp_path: Path) -> None:
    """The same cell under a second, unscaled measure: each fact must be read back through the
    measure that made it, not the first one with its field."""

    def second_column(spec: dict) -> None:
        shipped = next(x for x in spec["measures"] if x["field"] == "shipped_units")
        unscaled = {k: v for k, v in shipped.items() if k not in ("scale", "precision")}
        spec["measures"].insert(0, unscaled)

    m = _score(tmp_path, EditedTruth(lambda p: _edit_spec(p, "legacy_report_na", second_column)))
    assert m["metrics"]["provenance_accuracy"] == 1.0


def test_row_rejects_on_different_sheets_are_counted_apart(tmp_path: Path) -> None:
    """With no chart-of-accounts aliases every trial-balance row rejects: 12 rows x 20 sheets."""

    def no_accounts(project: Path) -> None:
        path = project / "crosswalks" / "coa.csv"
        path.write_text(path.read_text().splitlines()[0] + "\n")

    world = generate(DEFAULT_SEED, tmp_path / "world")
    loads, _ = run_mapper(EditedTruth(no_accounts), world, tmp_path / "proposed")
    real = sum(len(x.result.rejects) for x in loads if x.result is not None)
    m = score_rejects(load_truth(world), loads)
    assert m["rejects_total"] == real
    assert m["reject_precision"] < 0.05


def test_ambiguous_date_skips_a_day_that_equals_its_month(tmp_path: Path) -> None:
    world = generate(DEFAULT_SEED, tmp_path)
    log = [json.loads(x) for x in (world / "truth" / "messes.jsonl").read_text().splitlines()]
    rows = {int(m["detail"].split()[1]) for m in log if m["type"] == "ambiguous_date"}
    lines = (world / "sources" / SOURCE_FILES["C"]).read_text().splitlines()
    for n, line in enumerate(lines[1:], start=2):
        day, month, _ = line.split(";")[2].split("/")
        ambiguous = int(day) <= 12 and day != month
        assert (n in rows) == ambiguous, line


# --- the baseline mapper ----------------------------------------------------------------------


PLANTS = ["Detroit Stamping 07", "Detroit Stamping 17", "Toledo Assembly 12"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Detroit Stamping #7", "Detroit Stamping 07"),
        ("DET-07", "Detroit Stamping 07"),
        ("1012", "Toledo Assembly 12"),
        ("GRR-03", None),
        ("ZX4-07", None),
        ("Toldeo Assembly 12", "Toledo Assembly 12"),
    ],
)
def test_match_plant(raw: str, expected: str | None) -> None:
    assert match_plant(raw, PLANTS) == expected


def test_match_plant_falls_for_an_irregular_code() -> None:
    """Recorded, not wanted: the last two digits name the wrong plant. That's the floor."""
    assert match_plant("5112", PLANTS) == "Toledo Assembly 12"


def test_match_account() -> None:
    standard = ["4000 Net Revenue", "5000 Cost of Goods Sold", "6000 Selling Expense"]
    assert match_account("COGS", standard) == "5000 Cost of Goods Sold"
    assert match_account("Sales", standard) == "4000 Net Revenue"
    assert match_account("Turnover", standard) is None


def test_baseline_is_a_floor_with_room_above_it(tmp_path: Path) -> None:
    result = bench(DEFAULT_SEED, HeuristicMapper(), tmp_path)
    m = result["metrics"]
    assert m["facts_loaded"] > 0
    assert m["value_accuracy"] < 1.0
    assert m["false_merges"] > 0, "the irregular ERP codes are the trap for reading a code"
    assert result["errors"] == []


# --- the gate and the command line ------------------------------------------------------------


def test_committed_baseline_holds(tmp_path: Path) -> None:
    """The gate is a ratchet: no seed may regress; an improvement passes and is reported."""
    baseline = json.loads(DEFAULT_BASELINE.read_text())
    assert len(baseline["seeds"]) >= 3, "gate on several header wordings, not one"
    for seed in baseline["seeds"]:
        result = bench(int(seed), HeuristicMapper(), tmp_path / seed)
        assert check_gate(result, baseline) == [], seed


def _gate_baseline(**metrics) -> dict:
    return {"mapper": "heuristic", "seeds": {"1": metrics}}


def test_gate_catches_each_direction_of_regression() -> None:
    baseline = _gate_baseline(value_accuracy=0.9, false_merges=3, facts_loaded=100)

    def result(**metrics) -> dict:
        return {"seed": 1, "mapper": "heuristic", "metrics": metrics}

    assert check_gate(result(value_accuracy=0.9, false_merges=3, facts_loaded=1), baseline) == []
    assert check_gate(result(value_accuracy=0.95, false_merges=0, facts_loaded=1), baseline) == []
    assert check_gate(result(value_accuracy=0.89, false_merges=3), baseline)
    assert check_gate(result(value_accuracy=0.9, false_merges=4), baseline)
    assert check_gate(result(false_merges=3), baseline), "a missing metric is a regression"
    assert check_gate({**result(), "seed": 2}, baseline), "a seed the baseline lacks can't pass"
    better = improvements(result(value_accuracy=0.95, false_merges=3, facts_loaded=1), baseline)
    assert better == ["value_accuracy: 0.95 (baseline 0.9)"]


def test_self_check_fails_below_100_percent() -> None:
    result = {
        "metrics": {"value_accuracy": 0.99, "false_merges": 0, "facts_loaded": 5},
        "errors": [],
    }
    assert self_check(result) == ["value_accuracy: 0.99"]


def test_cli_truth_passes_and_writes_json(tmp_path: Path, capsys) -> None:
    out = tmp_path / "bench.json"
    assert cli.main(["wrangle", "bench", "--mapper", "truth", "--json", str(out)]) == 0
    assert "self-check" in capsys.readouterr().out
    assert json.loads(out.read_text())["metrics"]["value_accuracy"] == 1.0


def test_cli_gate_fails_on_a_regression(tmp_path: Path, capsys) -> None:
    baseline = json.loads(DEFAULT_BASELINE.read_text())
    seed = str(DEFAULT_SEED)
    baseline["seeds"] = {seed: {**baseline["seeds"][seed], "value_accuracy": 1.0}}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    assert cli.main(["wrangle", "bench", "--gate", str(path)]) == 1
    assert "value_accuracy" in capsys.readouterr().out


def test_cli_forwards_options_to_the_bench(monkeypatch) -> None:
    """`fde wrangle bench` takes bench.py's own options, `--gate` with no path included."""
    seen = []
    monkeypatch.setattr("evals.wrangle.bench.main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["wrangle", "bench", "--seed", "7", "13", "--gate"]) == 0
    assert seen == [["--seed", "7", "13", "--gate"]]


def test_failed_run_does_not_write_a_baseline(tmp_path: Path, capsys) -> None:
    baseline = json.loads(DEFAULT_BASELINE.read_text())
    seed = str(DEFAULT_SEED)
    baseline["seeds"] = {seed: {**baseline["seeds"][seed], "value_accuracy": 1.0}}
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(baseline))
    out = tmp_path / "new.json"
    assert main(["--gate", str(gate), "--write-baseline", str(out)]) == 1
    assert not out.exists(), "a regressed run must not become the floor"
    assert main(["--gate", str(gate), "--write-baseline", str(gate)]) == 2
    assert "same file" in capsys.readouterr().err


def test_write_baseline_keeps_other_seeds(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"mapper": "heuristic", "seeds": {"99": {"value_accuracy": 0.5}}}))
    assert main(["--seed", str(DEFAULT_SEED), "--write-baseline", str(path)]) == 0
    written = json.loads(path.read_text())
    assert set(written["seeds"]) == {str(DEFAULT_SEED), "99"}
    assert "rapidfuzz" in written["versions"]


def test_agent_mapper_is_not_built_yet(capsys) -> None:
    assert main(["--mapper", "agent"]) == 2
    assert "PR 7" in capsys.readouterr().err
