"""Multi-sheet specs (`source.sheets`, §5/§7.4) and possible-duplicate entity warnings
(`resolve.similar_warn` / `not_same`, §7.3), on small workbooks built here."""

from __future__ import annotations

import csv
from pathlib import Path

import pydantic
import pytest

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("rapidfuzz")

from fde_kit.wrangle.apply import (  # noqa: E402
    SpecMismatch,
    document_key,
    gate,
    run,
    unit_sheets,
    write_outputs,
)
from fde_kit.wrangle.labels import header_fingerprint  # noqa: E402
from fde_kit.wrangle.resolve import (  # noqa: E402
    SAME_BUT_PUNCTUATION,
    SIMILAR_SPELLING,
    similar_key,
    similar_names,
)
from fde_kit.wrangle.spec import MappingSpec, load_project_spec  # noqa: E402

DICTIONARY = """\
- field: sales
  definition: Sales.
  unit: USD
"""

SPEC = """\
spec: multi
version: 1
dictionary: ../dictionary.yaml
source:
{source}
entity:
  kind: store
  parts:
    - column: Store
{resolve}
period:
  unpivot:
    matching: "^Q\\\\d \\\\d{{4}}$"
    ops: [{{op: period, pattern: "Q{{q}} {{yyyy}}", grain: quarter}}]
measures:
  - field: sales
    ops: [{{op: number}}]
    nulls: ["(S)"]
"""

RESOLVE_OPEN = "      resolve: {policy: open}"


def _sheet(wb, name: str, year: int, rows: list[tuple], header: str = "Store") -> None:
    ws = wb.create_sheet(name)
    ws.append([f"Sales by store, {year}"])
    ws.append([header, *(f"Q{q} {year}" for q in (1, 2))])
    for r in rows:
        ws.append(list(r))


def _workbook(path: Path, sheets: dict[str, tuple[int, list[tuple]]], **headers: str) -> Path:
    """Sheets in the given (workbook) order; a title row, the header in row 2, data below."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, (year, rows) in sheets.items():
        _sheet(wb, name, year, rows, headers.get(name, "Store"))
    wb.save(path)
    return path


SHEETS = {
    "2024": (2024, [("North", 10, 11), ("South", 20, "(S)")]),
    "Notes": (2000, [("ignore me", 1, 2)]),
    "2023": (2023, [("North", 5, 6), ("South", 7, 8)]),
    "2022": (2022, [("North", 1, 2), ("East", 3, 4)]),
}


def _project(tmp_path: Path, source: str, resolve: str = RESOLVE_OPEN) -> Path:
    (tmp_path / "specs").mkdir(exist_ok=True)
    (tmp_path / "dictionary.yaml").write_text(DICTIONARY, encoding="utf-8")
    spec = tmp_path / "specs" / "multi.yaml"
    spec.write_text(SPEC.format(source=source, resolve=resolve), encoding="utf-8")
    return spec


@pytest.fixture
def book(tmp_path: Path) -> Path:
    return _workbook(tmp_path / "sales.xlsx", SHEETS)


MATCHING = '  sheets: {matching: "^\\\\d{4}$"}\n  header_rows: [2]'


# --- spec validation ------------------------------------------------------------------------


def _validate(source: dict) -> MappingSpec:
    return MappingSpec.model_validate(
        {
            "spec": "s",
            "dictionary": "d.yaml",
            "source": source,
            "entity": {"kind": "k", "parts": [{"column": "A"}]},
            "period": {"cell": {"ref": "A1", "ops": [{"op": "date"}]}},
            "measures": [{"field": "f", "column": "B", "ops": [{"op": "number"}]}],
        }
    )


@pytest.mark.parametrize(
    "source, message",
    [
        ({"sheet": "A", "sheets": ["B"]}, "at most one of `sheet` or `sheets`"),
        ({"sheets": []}, "source.sheets list is empty"),
        ({"sheets": ["A", "A"]}, "lists a sheet twice"),
        ({"sheets": {"matching": "("}}, "invalid regex"),
        ({"sheets": {"matching": "x", "exclude": ["a", "a"]}}, "exclude lists a sheet twice"),
        ({"sheets": {"matching": "x", "other": 1}}, "Extra inputs are not permitted"),
    ],
)
def test_sheets_validation(source: dict, message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        _validate(source)


def test_sheets_forms_parse() -> None:
    s = _validate({"sheets": ["2024", "2023"]}).source
    assert s.sheets == ["2024", "2023"] and s.multi_sheet
    s = _validate({"sheets": {"matching": "^\\d{4}$", "exclude": ["1999"]}}).source
    assert s.sheets.matching == "^\\d{4}$" and s.sheets.exclude == ["1999"]
    assert not _validate({"sheet": "A"}).source.multi_sheet


def test_sheets_is_hash_neutral_when_unset() -> None:
    payload = _validate({"sheet": "A"}).hash_payload()
    assert "sheets" not in payload["source"]
    payload = _validate({"sheets": ["A"]}).hash_payload()
    assert payload["source"]["sheets"] == ["A"]


def test_select_sheets_workbook_order_and_errors() -> None:
    names = ["2024", "Notes", "2023", "2022"]
    sel = _validate({"sheets": {"matching": "^\\d{4}$", "exclude": ["2023"]}}).source
    assert sel.select_sheets(names) == ["2024", "2022"]
    lst = _validate({"sheets": ["2022", "2024"]}).source
    assert lst.select_sheets(names) == ["2024", "2022"]  # workbook order, not list order
    with pytest.raises(ValueError, match=r"names \['1999'\], not in the workbook"):
        _validate({"sheets": ["1999"]}).source.select_sheets(names)
    with pytest.raises(ValueError, match=r"exclude names \['1999'\]"):
        _validate({"sheets": {"matching": "x", "exclude": ["1999"]}}).source.select_sheets(names)
    with pytest.raises(ValueError, match=r"no sheet matches '\^19'"):
        _validate({"sheets": {"matching": "^19"}}).source.select_sheets(names)


# --- apply ----------------------------------------------------------------------------------


def test_each_sheet_is_its_own_unit(tmp_path: Path, book: Path) -> None:
    spec = _project(tmp_path, MATCHING)
    result = run(spec, [book])
    assert [f.sheet for f in result.files] == ["2024", "2023", "2022"]  # workbook order
    assert [d.source for d in result.documents] == [
        f"multi/sales.xlsx#{s}" for s in ("2024", "2023", "2022")
    ]
    rep = result.report
    assert [(f["sheet"], f["rows_read"], f["null_skipped"]) for f in rep["files"]] == [
        ("2024", 2, {"(S)": 1}),
        ("2023", 2, {}),
        ("2022", 2, {}),
    ]
    assert rep["facts"] == 11 and rep["rejects"]["total"] == 0
    # Each fact's ref names its own sheet; "North" in three sheets is three documents' facts.
    by_sheet = {f.sheet: [r.cell for r in f.facts] for f in result.files}
    assert by_sheet["2023"] == ["2023!B3", "2023!C3", "2023!B4", "2023!C4"]
    assert all(r.fact.document_source.endswith("#2022") for r in result.files[2].facts)
    assert {e.canonical_name for e in result.entities} == {"North", "South", "East"}
    assert gate(result, draft=True) == []


def test_same_key_in_two_sheets_is_not_a_duplicate(tmp_path: Path) -> None:
    """Dedupe and within-load conflicts are per document: one sheet per document."""
    book = _workbook(
        tmp_path / "b.xlsx",
        {"2024": (2024, [("North", 1, 2)]), "2025": (2024, [("North", 1, 9)])},
    )
    result = run(_project(tmp_path, MATCHING), [book])
    assert result.report["facts"] == 4
    assert result.duplicates_dropped == 0 and result.conflicts == []


def test_list_form_and_exclude(tmp_path: Path, book: Path) -> None:
    spec = _project(tmp_path, '  sheets: ["2022", "2024"]\n  header_rows: [2]')
    assert [f.sheet for f in run(spec, [book]).files] == ["2024", "2022"]
    spec = _project(
        tmp_path, '  sheets: {matching: "^\\\\d{4}$", exclude: ["2024"]}\n  header_rows: [2]'
    )
    assert [f.sheet for f in run(spec, [book]).files] == ["2023", "2022"]
    assert unit_sheets(load_project_spec(spec), book) == ["2023", "2022"]


@pytest.mark.parametrize(
    "source, message",
    [
        ('  sheets: {matching: "^19"}', "no sheet matches '\\^19'"),
        ('  sheets: ["1999"]', "not in the workbook"),
        ('  sheets: {matching: "^2", exclude: ["1999"]}', "exclude names"),
    ],
)
def test_selection_mismatch(tmp_path: Path, book: Path, source: str, message: str) -> None:
    spec = _project(tmp_path, source + "\n  header_rows: [2]")
    with pytest.raises(SpecMismatch, match=message):
        run(spec, [book])


def test_one_bad_sheet_refuses_the_whole_load_and_is_named(tmp_path: Path) -> None:
    book = _workbook(tmp_path / "b.xlsx", SHEETS, **{"2023": "Shop"})
    with pytest.raises(SpecMismatch, match=r"sheet '2023': entity column 'Store' is not a col"):
        run(_project(tmp_path, MATCHING), [book])


def test_bounds_missing_in_one_sheet_name_it(tmp_path: Path) -> None:
    book = _workbook(
        tmp_path / "b.xlsx",
        {"2024": (2024, [("North", 1, 2), ("END", None, None)]), "2023": (2023, [("N", 1, 2)])},
    )
    spec = _project(tmp_path, MATCHING + '\n  stop_before: {column: Store, matches: "^END$"}')
    with pytest.raises(SpecMismatch, match=r"sheet '2023': source.stop_before: no row"):
        run(spec, [book])


def test_header_fingerprint_checked_per_sheet(tmp_path: Path) -> None:
    book = _workbook(
        tmp_path / "b.xlsx",
        {"2024": (2024, [("North", 1, 2)]), "2023": (2024, [("North", 3, 4)])},
    )
    fp = header_fingerprint(["Store", "Q1 2024", "Q2 2024"])
    spec = _project(tmp_path, MATCHING + f"\n  header_fingerprint: {fp}")
    assert run(spec, [book]).report["facts"] == 4  # both sheets have that header
    book2 = _workbook(tmp_path / "c.xlsx", SHEETS)  # 2023's labels say 2023
    with pytest.raises(SpecMismatch, match=r"c.xlsx sheet '2023': header_fingerprint is"):
        run(spec, [book2])
    assert run(spec, [book2], force=True).report["facts"] == 11


def test_document_key_without_sheet_collides(tmp_path: Path, book: Path) -> None:
    spec = _project(tmp_path, MATCHING + '\n  document_key: "{spec}/{file}"')
    with pytest.raises(SpecMismatch, match=r"sheet '2023': document_key .* renders 'multi/sa"):
        run(spec, [book])


def test_document_key_needs_the_sheet(tmp_path: Path, book: Path) -> None:
    loaded = load_project_spec(_project(tmp_path, MATCHING))
    assert document_key(loaded, book, "2022") == "multi/sales.xlsx#2022"
    with pytest.raises(SpecMismatch, match="several sheets"):
        document_key(loaded, book)


@pytest.mark.parametrize("source", ['  sheets: ["x"]', "  sheet: x"])
def test_csv_takes_neither_sheet_nor_sheets(tmp_path: Path, source: str) -> None:
    f = tmp_path / "sales.csv"
    f.write_text("Store,Q1 2024\nNorth,1\n", encoding="utf-8")
    spec = _project(tmp_path, source)
    with pytest.raises(SpecMismatch, match="is a CSV, whose only sheet is its file stem"):
        run(spec, [f])


def test_rejects_carry_their_sheet(tmp_path: Path) -> None:
    book = _workbook(
        tmp_path / "b.xlsx",
        {"2024": (2024, [("North", 1, "x")]), "2023": (2023, [("North", "y", 2)])},
    )
    result = run(_project(tmp_path, MATCHING), [book])
    assert [(r["sheet"], r["cell"]) for r in result.rejects] == [("2024", "C3"), ("2023", "B3")]


# --- similar_names --------------------------------------------------------------------------


def test_similar_key_folds_case_and_whitespace_only() -> None:
    assert similar_key("  Beer,  WINE\tand Liquor ") == "beer, wine and liquor"


def test_score_threshold() -> None:
    names = ["Men's clothing stores", "Women's clothing stores", "Shoe stores"]
    (p,) = similar_names(names, 90)
    assert (p.a, p.b, p.score, p.reason) == (
        "Men's clothing stores",
        "Women's clothing stores",
        95.5,
        SIMILAR_SPELLING,
    )
    assert similar_names(names, 96) == []
    assert similar_names(names, 0) == []  # 0 turns the score off


def test_ratio_does_not_flag_parent_and_child_categories() -> None:
    """The reason for `fuzz.ratio` over token_set_ratio: a subset name isn't a near-duplicate."""
    assert similar_names(["Clothing stores", "Men's clothing stores"], 90) == []


def test_norm_key_pairs_always_reported() -> None:
    names = ["Beer, wine, and liquor stores", "Beer, wine and liquor stores", "DeWitt County"]
    names.append("De Witt County")
    got = [(p.a, p.b, p.score, p.reason) for p in similar_names(names, 0)]
    assert got == [
        ("Beer, wine and liquor stores", "Beer, wine, and liquor stores", 100.0,
         SAME_BUT_PUNCTUATION),
        ("De Witt County", "DeWitt County", 100.0, SAME_BUT_PUNCTUATION),
    ]  # fmt: skip
    # With the score on as well, each pair is reported once, with the punctuation reason.
    assert [p.reason for p in similar_names(names, 90)] == [SAME_BUT_PUNCTUATION] * 2


def test_not_same_suppresses_either_order() -> None:
    names = ["Lee County", "Lea County", "Lake County"]
    assert [(p.a, p.b) for p in similar_names(names, 90)] == [("Lea County", "Lee County")]
    assert similar_names(names, 90, not_same=[("Lee County", "Lea County")]) == []
    assert similar_names(names, 90, not_same=[("Lea County", "Lee County")]) == []
    assert len(similar_names(names, 90, not_same=[("lee county", "Lea County")])) == 1  # exact


def test_sorted_by_score_then_names() -> None:
    names = ["Abcdefghij", "Abcdefghik", "Xyzxyzxyzq", "Xyzxyzxyzw", "Ab-cdefghij"]
    got = [(p.score, p.a, p.b) for p in similar_names(names, 85)]
    assert got == sorted(got, key=lambda t: (-t[0], t[1], t[2]))
    assert got[0] == (100.0, "Ab-cdefghij", "Abcdefghij")


def test_cdist_and_extract_iter_agree() -> None:
    pytest.importorskip("numpy")
    names = [f"{w} County" for w in ("Lee", "Lea", "Adams", "Adair", "Clay", "Clark", "Polk")]
    names += [f"Name {i:04d}" for i in range(1200)]  # more than one cdist chunk
    assert similar_names(names, 88, use_cdist=True) == similar_names(names, 88, use_cdist=False)


# --- warnings through apply -----------------------------------------------------------------

COMPOSITE = """\
spec: comp
version: 1
dictionary: ../dictionary.yaml
crosswalks:
  regions: {{path: ../regions.csv, kind: region}}
source:
  sheet: S
  header_rows: [1]
entity:
  kind: site
  parts:
    - column: Region
      resolve: {{crosswalk: regions}}
    - column: Town
      resolve: {{policy: open{town}}}
    - column: Line
      resolve: {{policy: open}}
  template: "{{0}}/{{1}}/{{2}}"
period:
  cell: {{ref: E1, ops: [{{op: period, format: "%Y", grain: year}}]}}
measures:
  - field: sales
    column: Sales
    ops: [{{op: number}}]
"""


def _composite(tmp_path: Path, town: str = "") -> Path:
    (tmp_path / "specs").mkdir(exist_ok=True)
    (tmp_path / "dictionary.yaml").write_text(DICTIONARY, encoding="utf-8")
    (tmp_path / "regions.csv").write_text(
        "alias,canonical\nNorth East,Northeast\nSouth East,Southeast\n", encoding="utf-8"
    )
    spec = tmp_path / "specs" / "comp.yaml"
    spec.write_text(COMPOSITE.format(town=town), encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S"
    for row in [
        ("Region", "Town", "Line", "Sales", "2024"),
        ("Northeast", "Lee", "Springfield A", 1),
        ("Southeast", "Lea", "Springfield B", 2),  # other region: never compared with Lee
        ("Northeast", "Lee", "Springfield A", 1),  # the same row again: first cells stay
        ("Northeast", "Le-e", "Line X", 3),  # same region as Lee: punctuation pair
        ("Southeast", "Le-e", "Line W", 6),  # Southeast's Le-e vs Lea scores 66.7: quiet
        ("Northeast", "Lee", "Springfield B", 5),  # under Northeast / Lee: near Springfield A
        ("Northeast", "Leeds", "Line Y", None),  # no fact: not compared
    ]:
        ws.append(list(row))
    wb.save(tmp_path / "s.xlsx")
    return spec


def test_warnings_compare_open_parts_not_composites(tmp_path: Path) -> None:
    result = run(_composite(tmp_path), [tmp_path / "s.xlsx"])
    sim = result.report["similar_entities"]
    assert sim == [
        {
            "part": "Town",
            "kind": "site",
            "a": "Le-e",
            "b": "Lee",
            "score": 100.0,
            "reason": SAME_BUT_PUNCTUATION,
            "cells": ["S!B5", "S!B2"],
            "within": ["Northeast"],
        },
        {
            "part": "Line",
            "kind": "site",
            "a": "Springfield A",
            "b": "Springfield B",
            "score": 92.3,
            "reason": SIMILAR_SPELLING,
            "cells": ["S!C2", "S!C7"],
            "within": ["Northeast", "Lee"],
        },
    ]
    # Region is a crosswalk part (Northeast/Southeast score 88.9 anyway): never compared.
    # Scopes are the parts to the left: Southeast's Lea and Le-e never meet Northeast's Lee,
    # and Springfield B under Southeast / Lea never meets Springfield A. "Leeds" has no fact.
    assert result.similar_entities == sim
    assert result.report["facts"] == 5 and result.duplicates_dropped == 1


def test_warnings_never_touch_counts_or_gate(tmp_path: Path) -> None:
    spec = _composite(tmp_path)
    warned = run(spec, [tmp_path / "s.xlsx"])
    quiet = run(_composite(tmp_path, ", similar_warn: 0, not_same: [[Lee, Le-e]]"),
                [tmp_path / "s.xlsx"])  # fmt: skip
    # Town: score off, and the punctuation pair reviewed; Line keeps the default 90.
    assert [d["part"] for d in quiet.report["similar_entities"]] == ["Line"]
    for key in ("facts", "duplicates_dropped", "rejects", "entities", "conflicts_within_load"):
        assert warned.report[key] == quiet.report[key], key
    assert gate(warned, draft=True) == gate(quiet, draft=True) == []


def test_similar_csv(tmp_path: Path) -> None:
    result = run(_composite(tmp_path), [tmp_path / "s.xlsx"])
    paths = write_outputs(result, tmp_path / "out")
    rows = list(csv.reader(paths["similar"].open(encoding="utf-8")))
    assert rows == [
        ["part", "kind", "within", "a", "b", "score", "reason", "cell_a", "cell_b"],
        ["Town", "site", "Northeast", "Le-e", "Lee", "100", SAME_BUT_PUNCTUATION, "S!B5",
         "S!B2"],
        ["Line", "site", "Northeast / Lee", "Springfield A", "Springfield B", "92.3",
         SIMILAR_SPELLING, "S!C2", "S!C7"],
    ]  # fmt: skip


@pytest.mark.parametrize(
    "resolve, message",
    [
        ({"not_same": [["a", "a"]]}, "paired with itself"),
        ({"not_same": [["a", "b"], ["b", "a"]]}, "listed twice"),
        ({"similar_warn": 101}, "less than or equal to 100"),
        ({"not_same": [["a"]]}, "at least 2 items|Field required"),
    ],
)
def test_resolve_warning_settings_validate(resolve: dict, message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        MappingSpec.model_validate(
            {
                "spec": "s",
                "dictionary": "d.yaml",
                "entity": {"kind": "k", "parts": [{"column": "A", "resolve": resolve}]},
                "period": {"cell": {"ref": "A1", "ops": [{"op": "date"}]}},
                "measures": [{"field": "f", "column": "B", "ops": [{"op": "number"}]}],
            }
        )


def test_resolve_warning_settings_hash_only_when_set() -> None:
    def payload(resolve: dict) -> dict:
        spec = MappingSpec.model_validate(
            {
                "spec": "s",
                "dictionary": "d.yaml",
                "entity": {"kind": "k", "parts": [{"column": "A", "resolve": resolve}]},
                "period": {"cell": {"ref": "A1", "ops": [{"op": "date"}]}},
                "measures": [{"field": "f", "column": "B", "ops": [{"op": "number"}]}],
            }
        )
        return spec.hash_payload()["entity"]["parts"][0]["resolve"]

    assert payload({}) == payload({"similar_warn": 90}) == payload({"not_same": []})
    assert "similar_warn" not in payload({}) and "not_same" not in payload({})
    assert payload({"similar_warn": 95})["similar_warn"] == 95
    assert payload({"not_same": [["a", "b"]]})["not_same"] == [["a", "b"]]
