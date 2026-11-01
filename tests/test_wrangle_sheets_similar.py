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
    SIMILAR_REPORTED,
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
    max_length_ratio,
    similar_key,
    similar_names,
)
from fde_kit.wrangle.spec import (  # noqa: E402
    MappingSpec,
    bundle_sha256,
    load_project_spec,
)

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


def _spec(resolve: dict, **over) -> MappingSpec:
    raw = {
        "spec": "s",
        "dictionary": "d.yaml",
        "entity": {"kind": "k", "parts": [{"column": "A", "resolve": resolve}]},
        "period": {"cell": {"ref": "A1", "ops": [{"op": "date"}]}},
        "measures": [{"field": "f", "column": "B", "ops": [{"op": "number"}]}],
    }
    raw.update(over)
    return MappingSpec.model_validate(raw)


def test_resolve_warning_settings_never_enter_the_hash() -> None:
    """They only shape warnings, and the bundle hash is in every document and source_ref:
    reviewing one pair into `not_same` must not re-version every document and fact."""

    def payload(resolve: dict) -> dict:
        return _spec(resolve).hash_payload()["entity"]["parts"][0]["resolve"]

    assert payload({}) == payload({"similar_warn": 90}) == payload({"not_same": []})
    assert payload({}) == payload({"similar_warn": 95, "not_same": [["a", "b"]]})
    assert "similar_warn" not in payload({}) and "not_same" not in payload({})


def test_warning_settings_leave_bundle_sha256_unchanged() -> None:
    files = {"d.yaml": "0" * 64}
    base = bundle_sha256(_spec({}), files)
    for resolve in ({"similar_warn": 0}, {"similar_warn": 99}, {"not_same": [["a", "b"]]}):
        assert bundle_sha256(_spec(resolve), files) == base, resolve
    # Every fact-affecting setting still changes it.
    changed = [
        _spec({}, entity={"kind": "k", "parts": [{"column": "A", "ops": [{"op": "text"}]}]}),
        _spec({"fuzzy": True}),
        _spec({"threshold": 80}),
        _spec({}, entity={"kind": "k2", "parts": [{"column": "A"}]}),
        _spec({}, entity={"kind": "k", "parts": [{"column": "A2"}]}),
        _spec({}, period={"cell": {"ref": "A2", "ops": [{"op": "date"}]}}),
        _spec({}, measures=[{"field": "f", "column": "C", "ops": [{"op": "number"}]}]),
    ]
    for spec in changed:
        assert bundle_sha256(spec, files) != base, spec
    assert bundle_sha256(_spec({}), {"d.yaml": "1" * 64}) != base


def test_reviewing_a_pair_keeps_documents_and_facts(tmp_path: Path) -> None:
    """Adding `not_same` / `similar_warn` keeps the bundle, every document text and every
    source_ref byte-identical; only the warnings change."""
    before = run(_composite(tmp_path), [tmp_path / "s.xlsx"])
    after = run(_composite(tmp_path, ", similar_warn: 0, not_same: [[Lee, Le-e]]"),
                [tmp_path / "s.xlsx"])  # fmt: skip
    assert before.loaded.bundle_sha256 == after.loaded.bundle_sha256
    assert [d.text for d in before.documents] == [d.text for d in after.documents]
    assert [f.row.source_ref for f in before.facts] == [f.row.source_ref for f in after.facts]
    assert before.report["similar_entities"] != after.report["similar_entities"]


# --- bounded warnings: digits-only pairs, the length window, the cap, the name limit ---------


def test_names_differing_only_in_digits_are_not_similar_spelling() -> None:
    names = ["Building 00001", "Building 00002", "Building 00010", "Unit 4", "Unit 5", "Q1 sales",
             "Q2 sales"]  # fmt: skip
    assert similar_names(names, 90) == []
    assert similar_names(names, 50) == []  # however low the threshold
    # A letter changed as well still fires; zeros alone are a punctuation pair (norm_key).
    got = similar_names(["Building 00001", "Bilding 00002", "Unit 4", "Unit 04"], 80)
    assert [(p.a, p.b, p.reason) for p in got] == [
        ("Unit 04", "Unit 4", SAME_BUT_PUNCTUATION),
        ("Bilding 00002", "Building 00001", SIMILAR_SPELLING),
    ]


def test_max_length_ratio_is_the_ratio_bound() -> None:
    from rapidfuzz import fuzz

    assert max_length_ratio(90) == pytest.approx(110 / 90)
    assert max_length_ratio(100) == 1.0
    # The bound is tight: a key and its own extension reach exactly the cutoff at the edge.
    base = "a" * 9
    assert fuzz.ratio(base, base + "b" * 2) >= 90 and 11 <= 9 * max_length_ratio(90)
    assert fuzz.ratio(base, base + "b" * 3) < 90 and 12 > 9 * max_length_ratio(90)


def _brute_force(
    keys: list[str], cutoff: float, groups: list[str] | None = None
) -> list[tuple[int, int]]:
    from rapidfuzz import fuzz

    return [
        (i, j)
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
        if fuzz.ratio(keys[i], keys[j]) >= cutoff and (groups is None or groups[i] != groups[j])
    ]


@pytest.mark.parametrize("cutoff", [60, 85, 90, 100])
@pytest.mark.parametrize("use_cdist", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_length_window_finds_every_pair(cutoff: float, use_cdist: bool, grouped: bool) -> None:
    """Skipping pairs outside the length window (and, with groups, a key's own group) loses
    nothing against scoring every pair."""
    if use_cdist:
        pytest.importorskip("numpy")
    import random

    from fde_kit.wrangle.resolve import _scored_pairs, _without_digits

    rng = random.Random(7)
    keys = ["", "ab", "abc", "abcd", "abc", "a1", "a2", "a10", "b1"]
    for _ in range(700):  # more than one cdist chunk, lengths 1..30
        keys.append("".join(rng.choice("abc 0123") for _ in range(rng.randint(1, 30))))
    groups = [_without_digits(k) for k in keys] if grouped else None
    got = [(i, j) for i, j, _ in _scored_pairs(keys, cutoff, use_cdist, groups)]
    assert got == _brute_force(keys, cutoff, groups)


def _many_names_project(tmp_path: Path, names: list[str], resolve: str = RESOLVE_OPEN) -> Path:
    _workbook(tmp_path / "many.xlsx", {"2024": (2024, [(n, 1, 2) for n in names])})
    return _project(tmp_path, '  sheet: "2024"\n  header_rows: [2]', resolve)


# 676 names; one changed letter (or a swap or shift of the pair) scores 90.9, so thousands of
# pairs fire.
ONE_LETTER_APART = [f"Store {a}{b}xyz" for a in "abcdefghijklmnopqrstuvwxyz" for b in
                    "abcdefghijklmnopqrstuvwxyz"]  # fmt: skip


def test_warnings_capped_with_true_total(tmp_path: Path) -> None:
    result = run(_many_names_project(tmp_path, ONE_LETTER_APART), [tmp_path / "many.xlsx"])
    rep = result.report
    keys = [similar_key(n) for n in sorted(ONE_LETTER_APART)]
    assert rep["similar_entities_total"] == len(_brute_force(keys, 90)) > 30_000
    assert len(rep["similar_entities"]) == SIMILAR_REPORTED == 200
    assert result.similar_entities == rep["similar_entities"]
    order = [(-d["score"], d["a"], d["b"]) for d in rep["similar_entities"]]
    assert order == sorted(order)
    assert rep["similar_entities"][0]["a"] == "Store aaxyz"  # the first in report order
    assert rep["similar_skipped"] == []
    rows = (write_outputs(result, tmp_path / "out")["similar"]).read_text().splitlines()
    assert len(rows) == 1 + 200


def test_numbered_series_does_not_flood(tmp_path: Path) -> None:
    """10k names like `Building 00001` used to give 651k warnings (166 MB report)."""
    names = [f"Building {i:05d}" for i in range(1, 2001)] + ["Bilding 00007"]
    result = run(_many_names_project(tmp_path, names), [tmp_path / "many.xlsx"])
    assert result.report["similar_entities_total"] == 1
    (d,) = result.report["similar_entities"]
    assert (d["a"], d["b"]) == ("Bilding 00007", "Building 00007")


def test_scored_half_skipped_over_the_name_limit(tmp_path: Path, monkeypatch) -> None:
    import fde_kit.wrangle.apply as apply_mod

    monkeypatch.setattr(apply_mod, "SIMILAR_MAX_SCORED_NAMES", 5)
    names = ["Lee County", "Lea County", "Le-e County", "Adams", "Clark", "Polk"]
    result = run(_many_names_project(tmp_path, names), [tmp_path / "many.xlsx"])
    rep = result.report
    assert rep["similar_skipped"] == [{"part": "Store", "names": 6}]
    # The norm_key half still runs; Lee/Lea (a scored pair) is not reported.
    assert [(d["a"], d["b"], d["reason"]) for d in rep["similar_entities"]] == [
        ("Le-e County", "Lee County", SAME_BUT_PUNCTUATION)
    ]
    monkeypatch.setattr(apply_mod, "SIMILAR_MAX_SCORED_NAMES", 6)
    again = run(_many_names_project(tmp_path, names), [tmp_path / "many.xlsx"])
    assert again.report["similar_skipped"] == []
    assert [(d["a"], d["b"]) for d in again.report["similar_entities"]] == [
        ("Le-e County", "Lee County"),
        ("Lea County", "Lee County"),
    ]


def test_scored_half_skipped_per_scope_names_within(tmp_path: Path, monkeypatch) -> None:
    import fde_kit.wrangle.apply as apply_mod

    monkeypatch.setattr(apply_mod, "SIMILAR_MAX_SCORED_NAMES", 1)
    result = run(_composite(tmp_path), [tmp_path / "s.xlsx"])
    assert result.report["similar_skipped"] == [
        {"part": "Town", "within": ["Northeast"], "names": 2},
        {"part": "Town", "within": ["Southeast"], "names": 2},
        {"part": "Line", "within": ["Northeast", "Lee"], "names": 2},
    ]
    # The punctuation pair under Northeast is still found.
    assert [d["reason"] for d in result.report["similar_entities"]] == [SAME_BUT_PUNCTUATION]


# --- warning cells name their file when a load has several -----------------------------------


def test_cells_carry_the_file_when_a_load_has_several(tmp_path: Path) -> None:
    east = _workbook(tmp_path / "east.xlsx", {"2024": (2024, [("Lee County", 1, 2)])})
    west = _workbook(tmp_path / "west.xlsx", {"2024": (2024, [("Lea County", 3, 4)])})
    spec = _project(tmp_path, '  sheet: "2024"\n  header_rows: [2]')
    (d,) = run(spec, [east, west]).report["similar_entities"]
    assert (d["a"], d["b"]) == ("Lea County", "Lee County")
    assert d["cells"] == ["west.xlsx#2024!A3", "east.xlsx#2024!A3"]
    # One file: cells stay `Sheet!A1`, as before.
    both = _workbook(tmp_path / "both.xlsx", {"2024": (2024, [("Lee County", 1, 2),
                                                               ("Lea County", 3, 4)])})  # fmt: skip
    (one,) = run(spec, [both]).report["similar_entities"]
    assert one["cells"] == ["2024!A4", "2024!A3"]


def test_cells_use_the_path_when_file_names_repeat(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = _workbook(tmp_path / "a" / "sales.xlsx", {"2024": (2024, [("Lee County", 1, 2)])})
    two = _workbook(tmp_path / "b" / "sales.xlsx", {"2024": (2024, [("Lea County", 3, 4)])})
    spec = _project(tmp_path, '  sheet: "2024"\n  header_rows: [2]')
    spec.write_text(
        spec.read_text().replace("source:\n", 'source:\n  document_key: "{spec}/{path}"\n'),
        encoding="utf-8",
    )
    (d,) = run(spec, [one, two]).report["similar_entities"]
    assert d["cells"] == [f"{two}#2024!A3", f"{one}#2024!A3"]
