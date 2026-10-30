"""Mapping spec: validation, project loading, bundle hash and approval (no database)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fde_kit.wrangle.spec import (
    DictEntry,
    MappingSpec,
    SpecError,
    approve,
    is_approved,
    load_project_spec,
)

DICTIONARY = """\
- field: demand_forecast_units
  definition: Forecast customer demand for the period, in units.
  unit: units
- field: plan_status
  definition: Planner's status label for the line.
"""

PLANTS = "alias,canonical\nDetroit,DET-07\nPlant 07,DET-07\nToledo,TOL-02\n"

SPEC = {
    "spec": "planner_forecast",
    "version": 1,
    "dictionary": "../dictionary.yaml",
    "crosswalks": {"plants": {"path": "../crosswalks/plants.csv", "kind": "plant"}},
    "source": {
        "sheet": "Plan",
        "header_rows": [4, 5],
        "data_starts": 6,
        "header_fingerprint": "9c1e2b7a55f0",
        "document_key": "planner-forecast#{sheet}",
        "skip_rows": [
            {"column": "Site", "matches": "(?i)^(sub)?total"},
            {"blank": ["Site"]},
        ],
    },
    "entity": {
        "kind": "plant_part",
        "parts": [
            {
                "column": "Site",
                "ops": [{"op": "text", "strip": True}],
                "resolve": {"crosswalk": "plants", "policy": "closed", "fuzzy": False},
            },
            {
                "column": "Part No",
                "ops": [{"op": "text", "strip": True, "case": "upper"}],
                "resolve": {"policy": "open"},
            },
        ],
        "template": "{0}/{1}",
    },
    "period": {
        "unpivot": {
            "columns": {"from": "Jan-26", "to": "Dec-26"},
            "ops": [{"op": "period", "format": "%b-%y", "grain": "month"}],
        }
    },
    "measures": [
        {
            "field": "demand_forecast_units",
            "ops": [{"op": "number", "thousands": ",", "parens_negative": True}],
            "scale": 1000,
        }
    ],
    "notes": 'Header says "(000s)".\n',
}


def write_project(root: Path, spec: dict | None = None, *, plants: str = PLANTS) -> Path:
    (root / "specs").mkdir(parents=True, exist_ok=True)
    (root / "crosswalks").mkdir(exist_ok=True)
    (root / "dictionary.yaml").write_text(DICTIONARY, encoding="utf-8")
    (root / "crosswalks" / "plants.csv").write_bytes(("﻿" + plants).encode("utf-8"))
    path = root / "specs" / "planner_forecast.yaml"
    path.write_text(
        "# Planner forecast, wide layout\n" + yaml.safe_dump(spec or SPEC, sort_keys=False),
        encoding="utf-8",
    )
    return path


def spec_with(**changes) -> dict:
    """SPEC deep-copied, with dotted-path keys replaced (value None deletes)."""
    out = copy.deepcopy(SPEC)
    for dotted, value in changes.items():
        node = out
        *parents, last = dotted.split("__")
        for p in parents:
            node = node[int(p)] if isinstance(node, list) else node[p]
        if value is None:
            del node[last]
        else:
            node[last] = value
    return out


# --- a valid project ------------------------------------------------------------------------


def test_valid_spec_loads(tmp_path):
    loaded = load_project_spec(write_project(tmp_path))
    spec = loaded.spec
    assert spec.spec == "planner_forecast"
    assert spec.source.first_data_row == 6
    assert spec.source.encoding == "utf-8-sig"
    assert len(spec.source.skip_rows) == 2
    assert spec.period.unpivot.columns.from_ == "Jan-26"
    assert spec.entity.render(["DET-07", "P-1"]) == "DET-07/P-1"
    assert spec.entity.parts[0].resolve.policy == "closed"

    assert loaded.dictionary["demand_forecast_units"] == DictEntry(
        "demand_forecast_units",
        "Forecast customer demand for the period, in units.",
        "units",
    )
    plants = loaded.crosswalks["plants"]
    assert plants.kind == "plant"
    assert plants.canonicals == {"DET-07", "TOL-02"}
    assert plants.mapping["Plant 07"] == "DET-07"
    assert plants.mapping["DET-07"] == "DET-07"  # canonical matches itself
    assert "alias" not in plants.mapping  # BOM stripped, header not data
    assert plants.aliases_for("DET-07") == ("Detroit", "Plant 07")

    assert set(loaded.files) == {"../dictionary.yaml", "../crosswalks/plants.csv"}
    assert len(loaded.bundle_sha256) == 64
    assert loaded.bundle_sha12 == loaded.bundle_sha256[:12]
    assert not loaded.approved


def test_effective_defaults(tmp_path):
    loaded = load_project_spec(write_project(tmp_path))
    m = loaded.spec.measures[0]
    assert m.effective_precision == 1000  # scale > 1
    assert loaded.unit_for(m) == "units"  # from the dictionary
    assert loaded.dictionary_entries() == [loaded.dictionary["demand_forecast_units"]]

    plain = spec_with(measures__0__scale=None, measures__0__unit="each")
    m = MappingSpec.model_validate(plain).measures[0]
    assert m.scale == 1 and m.effective_precision is None
    assert m.effective_unit(loaded.dictionary) == "each"

    explicit = spec_with(measures__0__precision=1)
    assert MappingSpec.model_validate(explicit).measures[0].effective_precision == 1


def test_defaults_without_optional_blocks():
    spec = spec_with(
        source={},
        crosswalks=None,
        entity={"kind": "plant", "parts": [{"column": "Site"}]},
        period={"column": {"name": "Month", "ops": [{"op": "date", "format": "%Y-%m"}]}},
        measures=[{"field": "demand_forecast_units", "column": "Qty", "ops": [{"op": "number"}]}],
    )
    m = MappingSpec.model_validate(spec)
    assert m.source.header_rows == [1] and m.source.first_data_row == 2
    assert m.source.document_key == "{path}#{sheet}"
    assert m.entity.template == "{0}"
    assert m.entity.parts[0].resolve.policy == "open"
    assert m.measures[0].empty == "skip"


def test_resolve_policy_defaults_closed_with_crosswalk():
    spec = spec_with(entity__parts__0__resolve={"crosswalk": "plants"})
    r = MappingSpec.model_validate(spec).entity.parts[0].resolve
    assert (r.policy, r.fuzzy, r.threshold) == ("closed", False, 92)


def test_period_cell_and_regex_op():
    spec = spec_with(
        period={
            "cell": {
                "ref": "B2",
                "ops": [
                    {"op": "regex", "pattern": "Period: (.+)"},
                    {"op": "period", "format": "%b %Y", "grain": "month"},
                ],
            }
        },
        measures__0__column="Qty",
    )
    assert MappingSpec.model_validate(spec).period.cell.ref == "B2"


def test_unpivot_matching_and_alias_round_trip():
    spec = spec_with(
        period__unpivot__columns=None, period__unpivot__matching=r"^[A-Z][a-z]{2}-\d{2}$"
    )
    m = MappingSpec.model_validate(spec)
    assert m.period.unpivot.matching
    dumped = MappingSpec.model_validate(SPEC).hash_payload()
    assert dumped["period"]["unpivot"]["columns"] == {"from": "Jan-26", "to": "Dec-26"}
    assert "approval" not in dumped


# --- validation errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source__colour": "red"}, "Extra inputs are not permitted"),
        ({"surprise": 1}, "Extra inputs are not permitted"),
        ({"entity__parts__0__resolve__fuzzy_ish": True}, "Extra inputs are not permitted"),
        ({"period__unpivot__columns__through": "x"}, "Extra inputs are not permitted"),
        ({"spec": "Planner-Forecast"}, "[a-z0-9_]+"),
        ({"measures__0__column": "Jan-26"}, "`column` must be omitted"),
        (
            {"period": {"column": {"name": "Month", "ops": []}}},
            "needs a `column`",
        ),
        (
            {
                "period": {
                    "column": {"name": "Month", "ops": []},
                    "unpivot": SPEC["period"]["unpivot"],
                }
            },
            "exactly one of column, cell, unpivot",
        ),
        ({"period": {}}, "exactly one of column, cell, unpivot"),
        (
            {"period__unpivot__matching": "^x$"},
            "exactly one of `columns` or `matching`",
        ),
        ({"period__unpivot__columns": None}, "exactly one of `columns` or `matching`"),
        ({"entity__template": "{0}"}, "exactly the placeholders {0}, {1}"),
        ({"entity__template": "{0}/{2}"}, "exactly the placeholders"),
        ({"entity__template": "{0}/{1}/{1:>3}"}, "no format specs"),
        ({"entity__template": "{site}/{1}"}, "exactly the placeholders"),
        ({"entity__parts__0__resolve__crosswalk": "sites"}, "not in `crosswalks`"),
        ({"entity__parts__0__ops": [{"op": "upper"}]}, "op must be one of"),
        ({"entity__parts__0__ops": [{"strip": True}]}, "op must be one of"),
        ({"measures__0__ops": [{"op": "regex", "pattern": "(a"}]}, "invalid regex"),
        ({"measures__0__ops": [{"op": "regex", "pattern": "a+"}]}, "exactly one group"),
        ({"measures__0__ops": [{"op": "regex", "pattern": "(a)(b)"}]}, "exactly one group"),
        ({"source__skip_rows": [{"column": "Site", "matches": "[unclosed"}]}, "invalid regex"),
        (
            {"period__unpivot__columns": None, "period__unpivot__matching": "(x"},
            "invalid regex",
        ),
        (
            {"measures__0__ops": [{"op": "text"}], "measures__0__precision": 1},
            "is text",
        ),
        (
            {
                "measures__0__ops": [{"op": "map", "values": {"Y": "yes"}}],
                "measures__0__precision": 5,
            },
            "is text",
        ),
        ({"measures__0__scale": 0}, "greater than 0"),
        ({"measures__0__empty": "zero"}, "'skip' or 'reject'"),
        ({"measures": []}, "at least 1"),
        ({"source__header_rows": [5, 4]}, "strictly increasing"),
        ({"source__data_starts": 5}, "after the last header row"),
        ({"source__document_key": "{path}#{tab}"}, "document_key may use only"),
        ({"source__header_fingerprint": "XYZ"}, "12 lowercase hex"),
        ({"source__delimiter": ";;"}, "single character"),
        ({"source__encoding": "utf-99"}, "unknown encoding"),
        ({"source__skip_rows": [{"blank": []}]}, "at least 1"),
        ({"source__skip_rows": [{"column": "Site"}]}, "matches"),
        ({"period": {"cell": {"ref": "b2", "ops": []}}}, "A1 cell reference"),
        ({"version": 2}, "1"),
    ],
)
def test_validation_errors(changes, message):
    with pytest.raises(ValidationError) as err:
        MappingSpec.model_validate(spec_with(**changes))
    assert message in str(err.value)


def test_text_measure_without_precision_is_fine():
    spec = spec_with(measures__0__ops=[{"op": "text"}], measures__0__scale=None)
    m = MappingSpec.model_validate(spec).measures[0]
    assert m.is_text and m.effective_precision is None


def test_measure_field_missing_from_dictionary(tmp_path):
    path = write_project(tmp_path, spec_with(measures__0__field="shipped_units"))
    with pytest.raises(SpecError, match="not in the dictionary"):
        load_project_spec(path)


def test_unknown_key_rejected_through_loader(tmp_path):
    path = write_project(tmp_path, spec_with(mesures=[]))
    with pytest.raises(ValidationError, match="mesures"):
        load_project_spec(path)


@pytest.mark.parametrize(
    ("plants", "message"),
    [
        ("name,canonical\nDetroit,DET-07\n", "header must be `alias,canonical`"),
        ("alias,canonical\nDetroit,DET-07\nDetroit,TOL-02\n", "maps to both"),
        ("alias,canonical\nTOL-02,DET-07\nToledo,TOL-02\n", "maps to both"),
        ("alias,canonical\nDetroit,\n", "empty alias or canonical"),
        ("alias,canonical\nDetroit,DET-07,extra\n", "expected 2 columns"),
    ],
)
def test_bad_crosswalk(tmp_path, plants, message):
    with pytest.raises(SpecError, match=message):
        load_project_spec(write_project(tmp_path, plants=plants))


def test_missing_crosswalk_file(tmp_path):
    path = write_project(tmp_path)
    (tmp_path / "crosswalks" / "plants.csv").unlink()
    with pytest.raises(SpecError, match="cannot read"):
        load_project_spec(path)


def test_bad_dictionary(tmp_path):
    path = write_project(tmp_path)
    (tmp_path / "dictionary.yaml").write_text("- field: x\n  definition: y\n  colour: red\n")
    with pytest.raises(SpecError, match="unknown keys"):
        load_project_spec(path)


# --- bundle hash ----------------------------------------------------------------------------


def test_bundle_hash_is_stable(tmp_path):
    a = load_project_spec(write_project(tmp_path / "a")).bundle_sha256
    b = load_project_spec(write_project(tmp_path / "b")).bundle_sha256
    assert a == b


def test_bundle_hash_changes_with_one_crosswalk_byte(tmp_path):
    before = load_project_spec(write_project(tmp_path / "a")).bundle_sha256
    after = load_project_spec(
        write_project(tmp_path / "b", plants=PLANTS.replace("TOL-02", "TOL-03"))
    ).bundle_sha256
    assert before != after
    # Even a byte that changes no mapping (a trailing newline) changes the hash.
    extra = load_project_spec(write_project(tmp_path / "c", plants=PLANTS + "\n")).bundle_sha256
    assert extra != before


def test_bundle_hash_changes_with_spec_and_dictionary(tmp_path):
    base = load_project_spec(write_project(tmp_path / "a")).bundle_sha256
    edited = load_project_spec(
        write_project(tmp_path / "b", spec_with(measures__0__scale=1))
    ).bundle_sha256
    assert edited != base

    path = write_project(tmp_path / "c")
    (tmp_path / "c" / "dictionary.yaml").write_text(DICTIONARY + "\n", encoding="utf-8")
    assert load_project_spec(path).bundle_sha256 != base


def test_bundle_hash_ignores_comments_layout_and_explicit_defaults(tmp_path):
    base = load_project_spec(write_project(tmp_path / "a")).bundle_sha256
    path = write_project(tmp_path / "b")
    path.write_text("# another comment\n\n" + path.read_text(), encoding="utf-8")
    assert load_project_spec(path).bundle_sha256 == base
    explicit = load_project_spec(
        write_project(tmp_path / "c", spec_with(source__encoding="utf-8-sig"))
    ).bundle_sha256
    assert explicit == base


def test_bundle_hash_ignores_approval_block(tmp_path):
    base = load_project_spec(write_project(tmp_path / "a")).bundle_sha256
    approval = {"by": "someone", "at": "2026-01-01T00:00:00Z", "bundle_sha256": "0" * 64}
    stamped = load_project_spec(write_project(tmp_path / "b", spec_with(approval=approval)))
    assert stamped.bundle_sha256 == base
    assert not is_approved(stamped)  # wrong hash in the stamp


# --- approval -------------------------------------------------------------------------------


def test_approve_round_trip(tmp_path):
    path = write_project(tmp_path)
    before = path.read_text()
    now = datetime(2026, 10, 30, 15, 2, 11, 999, tzinfo=UTC)
    loaded = approve(path, by="brooks", now=now)

    assert is_approved(loaded) and loaded.approved
    assert loaded.spec.approval.by == "brooks"
    assert loaded.spec.approval.at == datetime(2026, 10, 30, 15, 2, 11, tzinfo=UTC)
    text = path.read_text()
    assert text.startswith(before)  # comments and layout kept
    assert text.endswith(
        f'approval:\n  by: "brooks"\n  at: 2026-10-30T15:02:11Z\n'
        f'  bundle_sha256: "{loaded.bundle_sha256}"\n'
    )

    # Re-approving replaces the block instead of adding a second one.
    again = approve(path, by="alex", now=now)
    assert path.read_text().count("approval:") == 1
    assert again.spec.approval.by == "alex"
    assert again.bundle_sha256 == loaded.bundle_sha256
    assert is_approved(load_project_spec(path))


def test_approve_replaces_block_in_the_middle(tmp_path):
    path = write_project(tmp_path)
    text = path.read_text()
    marker = "measures:"
    # A malformed old stamp (YAML reads the all-digit hash as an int) doesn't block approve.
    stale = (
        "approval:\n  by: old\n  at: 2026-01-01T00:00:00Z\n\n  bundle_sha256: " + "0" * 64 + "\n"
    )
    path.write_text(text.replace(marker, stale + marker, 1), encoding="utf-8")
    loaded = approve(path, by="brooks")
    assert is_approved(loaded)
    out = path.read_text()
    assert out.count("approval:") == 1 and "old" not in out
    assert load_project_spec(path).spec.measures[0].field == "demand_forecast_units"


def test_stale_approval_after_crosswalk_edit(tmp_path):
    path = write_project(tmp_path)
    approve(path, by="brooks")
    assert is_approved(load_project_spec(path))
    csv_path = tmp_path / "crosswalks" / "plants.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"Motor City,DET-07\n")
    assert not is_approved(load_project_spec(path))


def test_approve_refuses_invalid_spec(tmp_path):
    path = write_project(tmp_path, spec_with(measures__0__field="shipped_units"))
    before = path.read_text()
    with pytest.raises(ValueError, match="not in the dictionary"):
        approve(path, by="brooks")
    assert path.read_text() == before

    with pytest.raises(ValueError, match="non-empty"):
        approve(write_project(tmp_path / "ok"), by="  ")
