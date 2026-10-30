"""Real-data regression: the wrangle load path against US government files (§10, `real/`).

The files in `tests/fixtures/wrangle/real/` are the oracle. Each spec in `real/project/specs/`
runs through `apply.run`, and every fact it produces is checked against an independent read of
the same file: openpyxl (`data_only=True`) or the `csv` module directly, never the wrangle
reader, labels or parse ops. Numbers are compared as `Decimal(str(cell value)) * scale`. The
oracle also decides which cells are data, which hold a placeholder (counted under
`null_skipped`), and which (entity, period) keys repeat (`duplicates_dropped`), so a load that
drops, adds or moves a single value fails. If a test here disagrees with a file, the code is
wrong, not the file.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from fde_kit.wrangle.apply import ApplyResult, gate, parse_source_ref, run  # noqa: E402

REAL = Path(__file__).parent / "fixtures" / "wrangle" / "real"
SPECS = REAL / "project" / "specs"

SOURCES = {
    "state_pop": "NST-EST2025-POP.xlsx",
    "eia_residential": "table_5_06_a.xlsx",
    "eia_transport": "table_5_06_a.xlsx",
    "county_pop": "co-est2025-alldata.slice.csv",
    "mrts_2024_nsa": "mrtssales92-present.xlsx",
    "mrts_2024_sa": "mrtssales92-present.xlsx",
    "generation_ct_ri": "generation_monthly.extract.xlsx",
}

MONTHS = {m: i for i, m in enumerate(calendar.month_abbr) if m}  # "Jan" -> 1, by the C locale


# --- the independent oracle -----------------------------------------------------------------


@dataclass
class Oracle:
    """What the file says the load must produce."""

    field: str
    unit: str
    scale: int = 1
    sheet: str = ""
    # (entity, start, end) -> [(cell ref, value)] in sheet order; the first one is the fact.
    cells: dict[tuple[str, date, date], list[tuple[str, Decimal]]] = field(default_factory=dict)
    nulls: Counter[str] = field(default_factory=Counter)
    period_from: dict[str, str] = field(default_factory=dict)
    """fact cell -> the period trail the report must show."""

    def add(self, entity: str, period: tuple[date, date], ref: str, value, trail: str) -> None:
        assert isinstance(value, int | float) and not isinstance(value, bool), (ref, value)
        amount = Decimal(str(value)) * self.scale
        self.cells.setdefault((entity, *period), []).append((ref, amount))
        self.period_from[ref] = trail

    @property
    def duplicates(self) -> int:
        return sum(len(v) - 1 for v in self.cells.values())


def _letter(col: int) -> str:
    out = ""
    while col:
        col, rem = divmod(col - 1, 26)
        out = chr(65 + rem) + out
    return out


def _month(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _year(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


def _sheet(name: str, sheet: str):
    wb = openpyxl.load_workbook(REAL / name, data_only=True)
    return wb[sheet]


def _label(*parts: object) -> str:
    """A joined header label as a person reads it: parts top to bottom, whitespace collapsed."""
    return " / ".join(" ".join(str(p).split()) for p in parts if p not in (None, ""))


def oracle_state_pop() -> Oracle:
    o = Oracle("population", "persons", sheet="NST-EST2025-POP")
    ws = _sheet(SOURCES["state_pop"], o.sheet)
    assert ws["C3"].value == "Population Estimate (as of July 1)"
    years = {c: ws.cell(4, c).value for c in range(3, 9)}  # C..H: 2020..2025, numeric cells
    assert list(years.values()) == [2020, 2021, 2022, 2023, 2024, 2025]
    for r in range(5, ws.max_row + 1):
        name, last = ws.cell(r, 1).value, ws.cell(r, 8).value
        if last is None:  # the blank row 61 and the notes below the table
            continue
        entity = name[1:] if name.startswith(".") else name  # `.Alabama`
        for c, y in years.items():
            ref = f"{_letter(c)}{r}"
            o.add(entity, _year(y), ref, ws.cell(r, c).value, _label(ws["C3"].value, y))
    return o


def _eia(sector: str, fld: str, placeholders: tuple[str, ...]) -> Oracle:
    o = Oracle(fld, "cents/kWh", sheet="Table_5_06_A")
    ws = _sheet(SOURCES["eia_residential"], o.sheet)
    cols = [c for c in range(1, ws.max_column + 1) if ws.cell(3, c).value == sector]
    (first,) = cols  # merged over two columns; the value is in the left one
    for c in (first, first + 1):
        month, year = ws.cell(4, c).value.split()
        period = _month(int(year), list(calendar.month_name).index(month))
        for r in range(5, ws.max_row + 1):
            entity, v = ws.cell(r, 1).value, ws.cell(r, c).value
            if v is None:  # the notes row 67 (merged across, so this cell is empty)
                continue
            if isinstance(v, str):
                assert v in placeholders, (f"{_letter(c)}{r}", v)
                o.nulls[v] += 1
                continue
            o.add(entity, period, f"{_letter(c)}{r}", v, _label(sector, ws.cell(4, c).value))
    return o


def oracle_eia_residential() -> Oracle:
    return _eia("Residential", "residential_price", ())


def oracle_eia_transport() -> Oracle:
    return _eia("Transportation", "transportation_price", ("--", "NM"))


def oracle_county_pop() -> Oracle:
    o = Oracle("population", "persons", sheet="co-est2025-alldata.slice")
    with (REAL / SOURCES["county_pop"]).open(encoding="latin-1", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    years = {header.index(f"POPESTIMATE{y}"): y for y in range(2020, 2026)}
    lv, st, cty = header.index("SUMLEV"), header.index("STNAME"), header.index("CTYNAME")
    for r, row in enumerate(rows[1:], start=2):
        if row[lv] == "040":  # a state row
            continue
        assert row[lv] == "050"
        for i, y in years.items():
            ref = f"{_letter(i + 1)}{r}"
            o.add(f"{row[st]}/{row[cty]}", _year(y), ref, int(row[i]), header[i])
    return o


def _mrts(fld: str, adjusted: bool) -> Oracle:
    o = Oracle(fld, "USD", scale=1_000_000, sheet="2024")
    ws = _sheet(SOURCES["mrts_2024_nsa"], o.sheet)
    assert "millions of dollars" in ws["A2"].value
    kinds = {r: ws.cell(r, 2).value for r in range(1, ws.max_row + 1)}
    nsa = next(r for r, v in kinds.items() if v == "NOT ADJUSTED")
    sa = next(r for r, v in kinds.items() if v == "ADJUSTED(2)")
    assert (nsa, sa) == (6, 71)
    months = {}
    for c in range(3, 15):  # C..N; O is TOTAL, not a month
        text = ws.cell(5, c).value  # "Jan. 2024", but "May 2024"
        abbr, year = text.replace(".", "").split()
        months[c] = (_month(int(year), MONTHS[abbr]), text)
    assert ws.cell(5, 15).value == "TOTAL"
    rows = range(sa + 1, ws.max_row + 1) if adjusted else range(nsa + 1, sa)
    for r in rows:
        kind = kinds[r]
        if not isinstance(kind, str) or not kind.strip() or ws.cell(r, 3).value is None:
            continue  # below the adjusted table: blank rows and footnotes in column A
        entity = re.sub(r"\(\d+\)$", "", kind)  # `GAFO(1)`
        for c, (period, text) in months.items():
            v = ws.cell(r, c).value
            if isinstance(v, str):
                assert v in ("(S)", "(NA)"), (f"{_letter(c)}{r}", v)
                o.nulls[v] += 1
                continue
            o.add(entity, period, f"{_letter(c)}{r}", v, text)
    return o


def oracle_mrts_2024_nsa() -> Oracle:
    return _mrts("retail_sales_nsa", adjusted=False)


def oracle_mrts_2024_sa() -> Oracle:
    return _mrts("retail_sales_sa", adjusted=True)


def oracle_generation_ct_ri() -> Oracle:
    o = Oracle("net_generation", "MWh", sheet="2024_Final")
    ws = _sheet(SOURCES["generation_ct_ri"], o.sheet)
    header = [ws.cell(5, c).value for c in range(1, 7)]
    assert header[:5] == ["YEAR", "MONTH", "STATE", "TYPE OF PRODUCER", "ENERGY SOURCE"]
    for r in range(6, ws.max_row + 1):
        year, month, state, producer, source, mwh = (ws.cell(r, c).value for c in range(1, 7))
        o.add(f"{state}/{producer}/{source}", _month(year, month), f"F{r}", mwh, f"A{r}+B{r}")
    return o


ORACLES = {name: globals()[f"oracle_{name}"] for name in SOURCES}


# --- running the specs ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def results() -> dict[str, ApplyResult]:
    return {name: run(SPECS / f"{name}.yaml", [REAL / f]) for name, f in SOURCES.items()}


@pytest.fixture(scope="module")
def oracles() -> dict[str, Oracle]:
    return {name: build() for name, build in ORACLES.items()}


def test_real_files_are_the_committed_bytes() -> None:
    """The oracle is only an oracle if nobody re-saved a file."""
    for line in (REAL / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((REAL / name).read_bytes()).hexdigest() == digest, name


# What each load must report. Rejects are 0 everywhere: every non-numeric cell in these
# files is either outside the data (titles, notes, the other table) or a declared placeholder.
COUNTS = {
    # name: (facts, duplicates_dropped, null_skipped)
    "state_pop": (342, 0, {}),
    "eia_residential": (124, 0, {}),
    "eia_transport": (75, 0, {"--": 48, "NM": 1}),
    "county_pop": (252, 0, {}),
    # NAICS 4522 and 45221 are both "Department stores" with equal values in all 12 months.
    "mrts_2024_nsa": (672, 12, {"(S)": 84}),
    "mrts_2024_sa": (432, 0, {"(S)": 24}),
    "generation_ct_ri": (816, 0, {}),
}


@pytest.mark.parametrize("name", list(SOURCES))
def test_report_counts(name: str, results: dict[str, ApplyResult]) -> None:
    rep = results[name].report
    facts, dups, nulls = COUNTS[name]
    assert rep["rejects"] == {"total": 0, "rate": 0.0, "by_reason": {}}, results[name].rejects[:5]
    assert rep["conflicts_within_load"] == []
    assert rep["facts"] == facts
    assert rep["duplicates_dropped"] == dups
    (f,) = rep["files"]
    assert f["null_skipped"] == nulls
    assert f["empty_skipped"] == 0
    assert gate(results[name], draft=True) == []


@pytest.mark.parametrize("name", list(SOURCES))
def test_every_fact_matches_the_file(
    name: str, results: dict[str, ApplyResult], oracles: dict[str, Oracle]
) -> None:
    result, o = results[name], oracles[name]
    (fr,) = result.files
    assert fr.sheet == o.sheet

    got: dict[tuple[str, date, date], tuple[str, object]] = {}
    for rec in fr.facts:
        row = rec.fact.row
        assert row.field == o.field and row.unit == o.unit
        assert row.precision == (o.scale if o.scale > 1 else None)
        ref = parse_source_ref(row.source_ref)
        assert ref["sheet"] == o.sheet and f"{ref['sheet']}!{ref['cell']}" == rec.cell
        assert ref["spec"] == name and ref["file"] == fr.file_sha256[:12]
        key = (row.entity, row.period_start, row.period_end)
        assert key not in got, f"{key} loaded twice ({got[key][0]}, {rec.cell})"
        got[key] = (ref["cell"], row.value)
        assert rec.period_from == o.period_from[ref["cell"]], rec.cell

    missing, extra = sorted(set(o.cells) - set(got)), sorted(set(got) - set(o.cells))
    assert not missing and not extra, f"missing {missing[:5]}, extra {extra[:5]}"
    for key, cells in o.cells.items():
        first_ref, want = cells[0]
        ref, value = got[key]
        assert ref == first_ref, key  # the first cell in sheet order is the one kept
        assert Decimal(value) == want, (key, ref, value, want)
        assert isinstance(value, int | Decimal) and not isinstance(value, bool)
        # Repeats of a key in the file carry the same value (else it would be a conflict).
        assert all(v == want for _, v in cells[1:]), (key, cells)
    assert result.duplicates_dropped == o.duplicates
    assert fr.null_skipped == dict(o.nulls)


# --- anchors checked by hand against the published tables ---------------------------------


def _facts(result: ApplyResult) -> dict[tuple[str, date], object]:
    return {(f.row.entity, f.row.period_start): f.row.value for f in result.facts}


def test_anchor_state_population(results: dict[str, ApplyResult]) -> None:
    facts = _facts(results["state_pop"])
    assert facts[("Connecticut", date(2025, 1, 1))] == 3_688_496
    assert facts[("United States", date(2025, 1, 1))] == 341_784_857
    assert facts[("Puerto Rico", date(2020, 1, 1))] == 3_281_591  # below the blank row 61
    assert not any(e.startswith(".") for e, _ in facts)
    assert len({e for e, _ in facts}) == 57  # US, 4 regions, 50 states, DC, PR


def test_anchor_county_latin1(results: dict[str, ApplyResult]) -> None:
    facts = _facts(results["county_pop"])
    assert facts[("New Mexico/Doña Ana County", date(2025, 1, 1))] == 229_091
    assert facts[("New Mexico/Doña Ana County", date(2020, 1, 1))] == 220_059
    names = {e for e, _ in facts}
    assert not any("Ã" in n or "�" in n for n in names)  # no mojibake
    assert "Connecticut/Connecticut" not in names and "New Mexico/New Mexico" not in names
    assert len(names) == 42  # 9 CT planning regions + 33 NM counties


def test_anchor_mrts_nsa_and_sa_differ(results: dict[str, ApplyResult]) -> None:
    nsa, sa = _facts(results["mrts_2024_nsa"]), _facts(results["mrts_2024_sa"])
    total = "Retail and food services sales, total"
    assert nsa[(total, date(2024, 1, 1))] == 603_091_000_000
    assert sa[(total, date(2024, 1, 1))] == 660_602_000_000
    assert nsa[(total, date(2024, 5, 1))] == 705_583_000_000  # the "May 2024" label
    assert sa[(total, date(2024, 12, 1))] == 691_794_000_000
    assert nsa[("GAFO", date(2024, 1, 1))] == 108_234_000_000
    assert ("Home furnishings stores", date(2024, 1, 1)) not in nsa  # (S)
    assert ("Department stores", date(2024, 1, 1)) in nsa
    # The adjusted table is shorter: it has no "Used car dealers" line.
    assert ("Used car dealers", date(2024, 1, 1)) in nsa
    assert ("Used car dealers", date(2024, 1, 1)) not in sa


def test_anchor_eia_prices(results: dict[str, ApplyResult]) -> None:
    res = _facts(results["eia_residential"])
    assert res[("Connecticut", date(2026, 8, 1))] == Decimal("24.51")
    assert res[("U.S. Total", date(2025, 8, 1))] == Decimal("17.57")
    tr = _facts(results["eia_transport"])
    assert ("Connecticut", date(2026, 8, 1)) not in tr  # NM
    assert tr[("Connecticut", date(2025, 8, 1))] == Decimal("18.67")
    assert ("Maine", date(2025, 8, 1)) not in tr  # --
    assert tr[("Rhode Island", date(2026, 8, 1))] == Decimal("22.58")


def test_anchor_generation(results: dict[str, ApplyResult]) -> None:
    facts = _facts(results["generation_ct_ri"])
    total = "CT/Total Electric Power Industry/Total"
    assert facts[(total, date(2024, 1, 1))] == 3_532_749
    assert facts[("CT/Combined Heat and Power, Commercial Power/Petroleum", date(2024, 1, 1))] == (
        -13_887
    )
    assert {d.month for e, d in facts if e == total} == set(range(1, 13))
    rec = results["generation_ct_ri"].files[0].facts[0]
    assert (rec.cell, rec.period_cells) == ("2024_Final!F6", ("A6", "B6"))


# --- without the features, the same files don't load ----------------------------------------


@pytest.fixture
def project_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(REAL / "project", dst)
    return dst


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_without_nulls_or_format_list_mrts_rejects(project_copy: Path) -> None:
    spec = project_copy / "specs" / "mrts_2024_nsa.yaml"
    _edit(spec, 'format: ["%b. %Y", "%b %Y"]', 'format: "%b. %Y"')
    _edit(spec, '    nulls: ["(S)", "(NA)"]\n', "")
    result = run(spec, [REAL / SOURCES["mrts_2024_nsa"]])
    by_reason = result.report["rejects"]["by_reason"]
    # Column G ("May 2024") fails its period for every non-placeholder cell; (S) cells fail
    # the number op (parentheses) in the other 11 months.
    assert by_reason == {"number: parentheses not allowed": 77, "period: unparseable": 64}
    assert {r["cell"][0] for r in result.rejects if r["stage"] == "period"} == {"G"}


def test_without_sections_mrts_tables_conflict(project_copy: Path) -> None:
    spec = project_copy / "specs" / "mrts_2024_nsa.yaml"
    _edit(spec, '  start_after: {column: Kind of Business, matches: "^NOT ADJUSTED$"}\n', "")
    _edit(
        spec,
        '  stop_before: {column: Kind of Business, matches: "^ADJUSTED"}       '
        '# "ADJUSTED(2)", row 71\n',
        '  skip_rows: [{blank: ["Jan. 2024"]}]\n',
    )
    result = run(spec, [REAL / SOURCES["mrts_2024_nsa"]])
    # The adjusted table's values load under the same keys: one conflict per line and month.
    assert len(result.report["conflicts_within_load"]) > 300
