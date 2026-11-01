"""Grid reading (§7.1): coordinates, kinds, merged ranges, formulas without cached values, CSV."""

from __future__ import annotations

import datetime as dt
import hashlib

import openpyxl
import pytest

from fde_kit.wrangle.parse import ParseError, run_ops
from fde_kit.wrangle.read import (
    FORMULA_NO_VALUE,
    FORMULA_NOT_CALCULATED,
    Cell,
    XlsxBook,
    col_letter,
    col_number,
    parse_ref,
    read_failure_reason,
    read_grid,
)


@pytest.mark.parametrize(
    ("n", "letters"),
    [(1, "A"), (7, "G"), (26, "Z"), (27, "AA"), (52, "AZ"), (53, "BA"), (702, "ZZ"), (703, "AAA")],
)
def test_col_letter_round_trip(n, letters):
    assert col_letter(n) == letters
    assert col_number(letters) == n


def test_col_letter_rejects_zero():
    with pytest.raises(ValueError):
        col_letter(0)


@pytest.mark.parametrize(
    ("ref", "want"),
    [("G42", (42, 7)), ("A1", (1, 1)), ("AA10", (10, 27)), ("$B$3", (3, 2)), ("c5", (5, 3))],
)
def test_parse_ref(ref, want):
    assert parse_ref(ref) == want
    row, col = want
    assert Cell(row, col, "", "empty").ref == ref.replace("$", "").upper()


@pytest.mark.parametrize("ref", ["", "42", "G", "G0", "1G", "Plan!G42", "G42:H43"])
def test_parse_ref_rejects(ref):
    with pytest.raises(ValueError):
        parse_ref(ref)


# --------------------------------------------------------------------------------- xlsx


@pytest.fixture
def plan_xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Plan"
    ws["A1"] = "Planner forecast (000s)"
    ws.merge_cells("B2:D2")
    ws["B2"] = "2026"
    ws["A3"] = "Site"
    ws["B3"] = "Jan-26"
    ws["C3"] = "Feb-26"
    ws["D3"] = "Total"
    ws["A4"] = "DET-07"
    ws["B4"] = 1234
    ws["C4"] = 0.1
    ws["D4"] = "=B4+C4"  # openpyxl never writes a cached value
    ws["A5"] = "GRY-01"
    ws["B5"] = "#N/A"
    ws["C5"] = "#DIV/0!"
    ws["D5"] = -1234.5
    ws["A6"] = dt.datetime(2026, 3, 4)
    ws["B6"] = dt.datetime(2026, 3, 4, 13, 45)
    ws["C6"] = dt.date(2026, 1, 31)
    ws["D6"] = True
    ws["E7"] = 1e20
    ws["F7"] = 1.5e-7
    ws["A7"] = "   "
    other = wb.create_sheet("Notes")
    other["A1"] = "hello"
    path = tmp_path / "forecast.xlsx"
    wb.save(path)
    return path


def test_xlsx_cells_and_kinds(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    assert g.sheet == "Plan"
    assert g.path == str(plan_xlsx)
    assert g.file_sha256 == hashlib.sha256(plan_xlsx.read_bytes()).hexdigest()

    def at(ref):
        c = g.cell(*parse_ref(ref))
        assert c.ref == ref
        return c.raw, c.kind

    assert at("A1") == ("Planner forecast (000s)", "text")
    assert at("A3") == ("Site", "text")
    assert at("B4") == ("1234", "number")
    assert at("C4") == ("0.1", "number")
    assert at("D5") == ("-1234.5", "number")
    assert at("E7") == ("100000000000000000000", "number")
    assert at("F7") == ("0.00000015", "number")
    assert at("B5") == ("#N/A", "error")
    assert at("C5") == ("#DIV/0!", "error")
    assert at("A6") == ("2026-03-04", "date")
    assert at("B6") == ("2026-03-04T13:45:00", "date")
    assert at("C6") == ("2026-01-31", "date")
    assert at("D6") == ("true", "bool")
    assert at("A7")[1] == "empty"
    assert at("A2") == ("", "empty")


def test_formula_without_cached_value_is_never_zero(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    c = g.cell(4, 4)
    assert (c.ref, c.kind, c.raw) == ("D4", "formula_no_value", "")
    with pytest.raises(ParseError):
        run_ops(c.raw, [{"op": "number"}])


def test_merged_ranges_recorded_not_filled(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    assert g.merged == [(2, 2, 2, 4)]
    assert g.cell(2, 2).raw == "2026"
    assert g.cell(2, 3).kind == "empty"
    assert g.cell(2, 4).kind == "empty"


def test_rows_are_rectangular_and_coordinates_match(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    assert len(g.rows) == 7
    assert {len(r) for r in g.rows} == {6}
    for i, row in enumerate(g.rows, start=1):
        for j, c in enumerate(row, start=1):
            assert (c.row, c.col) == (i, j)
    assert g.cell(100, 100) == Cell(100, 100, "", "empty")


def test_xlsx_numbers_and_dates_parse_with_ops(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    assert run_ops(g.cell(4, 2).raw, [{"op": "number"}]) == 1234
    assert str(run_ops(g.cell(4, 3).raw, [{"op": "number"}])) == "0.1"
    assert run_ops(g.cell(7, 5).raw, [{"op": "number"}]) == 10**20
    assert run_ops(g.cell(6, 1).raw, [{"op": "date", "format": "%Y-%m-%d"}]) == dt.date(2026, 3, 4)


def test_default_sheet_is_first_and_other_sheet_readable(plan_xlsx):
    assert read_grid(plan_xlsx).sheet == "Plan"
    notes = read_grid(plan_xlsx, "Notes")
    assert notes.rows == [[Cell(1, 1, "hello", "text")]]
    assert notes.merged == []


def test_missing_sheet_lists_available(plan_xlsx):
    with pytest.raises(ValueError, match=r"no sheet 'Forecast'.*'Plan', 'Notes'"):
        read_grid(plan_xlsx, "Forecast")


def test_cached_formula_value_is_read(tmp_path):
    """A formula whose cached value is present reads as that value (as LibreOffice saves it).

    openpyxl can't write cached values, so patch one into the sheet XML by hand, and drop the
    `fullCalcOnLoad` flag openpyxl sets (a file LibreOffice or Excel saved doesn't have it)."""
    import re
    import zipfile

    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = 2
    ws["A2"] = "=A1*3"
    ws["A3"] = "=1/0"
    src = tmp_path / "src.xlsx"
    wb.save(src)
    dst = tmp_path / "cached.xlsx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                xml = data.decode()
                xml = re.sub(r"<f>A1\*3</f>(<v\s*/>|<v></v>)?", "<f>A1*3</f><v>6</v>", xml)
                xml = xml.replace('<c r="A3"', '<c r="A3" t="e"')
                xml = re.sub(r"<f>1/0</f>(<v\s*/>|<v></v>)?", "<f>1/0</f><v>#DIV/0!</v>", xml)
                data = xml.encode()
            elif item.filename == "xl/workbook.xml":
                data = _without_full_calc(data)
            zout.writestr(item, data)
    g = read_grid(dst)
    assert (g.cell(2, 1).raw, g.cell(2, 1).kind) == ("6", "number")
    assert (g.cell(3, 1).raw, g.cell(3, 1).kind) == ("#DIV/0!", "error")


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="unsupported"):
        read_grid(p)


# ---------------------------------------------------------------------------------- csv


def test_csv_with_bom_and_comma(tmp_path):
    p = tmp_path / "erp_shipments.csv"
    p.write_bytes('Site,Qty,Date\nDET-07,"1,234",04/03/2026\nGRY-01,,\n'.encode("utf-8-sig"))
    g = read_grid(p)
    assert g.sheet == "erp_shipments"
    assert g.merged == []
    assert g.file_sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert g.cell(1, 1) == Cell(1, 1, "Site", "text")  # BOM stripped
    assert g.cell(2, 2) == Cell(2, 2, "1,234", "text")
    assert g.cell(3, 2) == Cell(3, 2, "", "empty")
    assert {c.kind for r in g.rows for c in r} <= {"text", "empty"}
    assert run_ops(g.cell(2, 2).raw, [{"op": "number"}]) == 1234


def test_csv_semicolon_sniffed(tmp_path):
    p = tmp_path / "legacy_report.csv"
    p.write_text(
        "Site;Datum;Menge\nDET-07;04/03/2026;1.234,5\nGRY-01;05/03/2026;(12,0)\n",
        encoding="utf-8",
    )
    g = read_grid(p)
    assert [c.raw for c in g.rows[0]] == ["Site", "Datum", "Menge"]
    assert g.cell(2, 3).raw == "1.234,5"
    ops = [{"op": "number", "thousands": ".", "decimal": ",", "parens_negative": True}]
    assert str(run_ops(g.cell(3, 3).raw, ops)) == "-12.0"


@pytest.mark.parametrize(("delim", "name"), [("\t", "tab"), ("|", "pipe")])
def test_csv_other_delimiters_sniffed(tmp_path, delim, name):
    p = tmp_path / f"{name}.csv"
    p.write_text(delim.join(["a", "b", "c"]) + "\n" + delim.join(["1", "2", "3"]) + "\n")
    g = read_grid(p)
    assert [c.raw for c in g.rows[1]] == ["1", "2", "3"]


def test_csv_explicit_delimiter_and_encoding(tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes("Bez;Wert\nMünchen;3\n".encode("latin-1"))
    g = read_grid(p, encoding="latin-1", delimiter=";")
    assert g.cell(2, 1).raw == "München"


def test_csv_single_column_falls_back_to_comma(tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("value\n1\n2\n")
    g = read_grid(p)
    assert [r[0].raw for r in g.rows] == ["value", "1", "2"]


def test_csv_ragged_rows_padded_and_blank_lines_keep_row_numbers(tmp_path):
    p = tmp_path / "ragged.csv"
    p.write_text("a,b,c\n1\n\n4,5,6,7\n\n\n")
    g = read_grid(p)
    assert len(g.rows) == 4  # trailing blank lines trimmed
    assert {len(r) for r in g.rows} == {4}
    assert g.cell(3, 1).kind == "empty"
    assert g.cell(4, 4) == Cell(4, 4, "7", "text")
    assert g.cell(4, 4).ref == "D4"


def test_csv_sheet_must_match_stem(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a\n1\n")
    assert read_grid(p, "data").sheet == "data"
    with pytest.raises(ValueError, match="CSV"):
        read_grid(p, "Plan")


def test_csv_wrong_encoding_is_a_clear_error(tmp_path):
    p = tmp_path / "counties.csv"
    p.write_bytes("STNAME,CTYNAME\nNew Mexico,Doña Ana County\n".encode("latin-1"))
    offset = p.read_bytes().index(b"\xf1")
    with pytest.raises(ValueError) as e:
        read_grid(p)
    msg = str(e.value)
    assert str(p) in msg and f"byte offset {offset}" in msg and "0xf1" in msg
    assert "source.encoding" in msg and "latin-1" in msg and "cp1252" in msg
    assert isinstance(e.value.__cause__, UnicodeDecodeError)
    assert read_grid(p, encoding="latin-1").cell(2, 2).raw == "Doña Ana County"


def test_csv_unknown_encoding(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n")
    with pytest.raises(ValueError, match="unknown encoding 'no-such-codec'"):
        read_grid(p, encoding="no-such-codec")


def test_xlsx_merged_hidden_values_dropped_and_far_empty_cells_ignored(tmp_path):
    """Only a merged range's top-left value is kept (as openpyxl's full model does), and an
    empty but styled cell far out doesn't widen or lengthen the grid."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "Title"
    ws["B1"] = "hidden under the merge"
    ws.merge_cells("A1:C1")
    ws["A2"] = 1
    ws["Z500"].number_format = "0.00"  # styled, no value
    ws["AA2"] = "   "  # whitespace only: empty
    path = tmp_path / "m.xlsx"
    wb.save(path)
    g = read_grid(path)
    assert g.merged == [(1, 1, 1, 3)]
    assert g.cell(1, 2) == Cell(1, 2, "", "empty")
    assert len(g.rows) == 2 and {len(r) for r in g.rows} == {1}  # trimmed to the values


def test_xlsx_reads_only_the_requested_sheet(tmp_path, monkeypatch):
    """The full worksheet model is never built (it took 2.8 GB on a 19 MB workbook)."""
    real = openpyxl.load_workbook
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(openpyxl, "load_workbook", spy)
    wb = openpyxl.Workbook()
    wb.active["A1"] = "x"
    path = tmp_path / "s.xlsx"
    wb.save(path)
    read_grid(path)
    assert calls == [{"read_only": True, "data_only": True}]


def _without_full_calc(workbook_xml: bytes) -> bytes:
    """openpyxl's `<calcPr calcId=".." fullCalcOnLoad="1"/>` without the flag, as Excel saves it."""
    import re

    out, n = re.subn(rb'\s*fullCalcOnLoad="(1|true)"', b"", workbook_xml)
    assert n == 1, "openpyxl no longer writes fullCalcOnLoad; revisit these tests"
    return out


def _patch_xlsx(src, dst, replace):
    """Copy an xlsx, with `replace[member] = new bytes or None (drop)` and renamed members."""
    import zipfile

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = replace.get(item.filename, zin.read(item.filename))
            zout.writestr(item.filename, data)


def test_shared_formulas_without_cached_values(tmp_path):
    """Excel writes a filled-down formula once (`<f t="shared" ref=.. si="0">`) and the other
    cells as an empty `<f t="shared" si="0"/>`. With no `<v>`, or an empty one, each such cell
    is `formula_no_value`, never empty; one with a cached `<v>` reads as that value."""
    wb = openpyxl.Workbook()
    wb.active.title = "Calc"
    src = tmp_path / "src.xlsx"
    wb.save(src)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = (
        f'<worksheet xmlns="{ns}"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>x2</t></is></c>'
        "</row>"
        '<row r="2"><c r="A2"><v>1</v></c>'
        '<c r="B2"><f t="shared" ref="B2:B5" si="0">A2*2</f></c></row>'
        '<row r="3"><c r="A3"><v>2</v></c><c r="B3"><f t="shared" si="0"/></c></row>'
        '<row r="4"><c r="A4"><v>3</v></c><c r="B4"><f t="shared" si="0"/><v></v></c></row>'
        '<row r="5"><c r="A5"><v>4</v></c><c r="B5"><f t="shared" si="0"/><v>8</v></c></row>'
        # no `r` on the row or its cells: numbered on as openpyxl does (row 6, A then B)
        '<row><c><v>5</v></c><c><f t="shared" si="0"/></c></row>'
        "</sheetData></worksheet>"
    )
    strings = f'<sst xmlns="{ns}" count="1" uniqueCount="1"><si><t>n</t></si></sst>'
    import zipfile

    dst = tmp_path / "shared.xlsx"
    with zipfile.ZipFile(src) as z:
        workbook = _without_full_calc(z.read("xl/workbook.xml"))
    _patch_xlsx(src, dst, {"xl/worksheets/sheet1.xml": sheet.encode(), "xl/workbook.xml": workbook})
    # add a shared-strings part the workbook relationships point to
    with zipfile.ZipFile(dst) as z:
        rels = z.read("xl/_rels/workbook.xml.rels").decode()
        types = z.read("[Content_Types].xml").decode()
    rels = rels.replace(
        "</Relationships>",
        '<Relationship Id="rIdSS" Target="sharedStrings.xml" Type="http://schemas.openxmlformats'
        '.org/officeDocument/2006/relationships/sharedStrings"/></Relationships>',
    )
    types = types.replace(
        "</Types>",
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats'
        '-officedocument.spreadsheetml.sharedStrings+xml"/></Types>',
    )
    dst2 = tmp_path / "shared2.xlsx"
    _patch_xlsx(
        dst,
        dst2,
        {"xl/_rels/workbook.xml.rels": rels.encode(), "[Content_Types].xml": types.encode()},
    )
    with zipfile.ZipFile(dst2, "a") as z:
        z.writestr("xl/sharedStrings.xml", strings)

    g = read_grid(dst2, "Calc")
    got = [[(c.raw, c.kind) for c in r] for r in g.rows]
    fnv = ("", "formula_no_value")
    assert got == [
        [("n", "text"), ("x2", "text")],
        [("1", "number"), fnv],
        [("2", "number"), fnv],
        [("3", "number"), fnv],
        [("4", "number"), ("8", "number")],
        [("5", "number"), fnv],
    ]


def test_sheet_xml_found_through_relationships_not_file_order(tmp_path):
    """Sheet files needn't be named in sheet order (Excel keeps `sheet3.xml` for a sheet moved
    to the front). The merged ranges and formula cells come from the sheet the name maps to."""
    wb = openpyxl.Workbook()
    first = wb.active
    first.title = "First"
    first["A1"] = "first"
    first.merge_cells("A1:B1")
    first["A2"] = "=1+1"
    wb.create_sheet("Second")["A1"] = "second"
    third = wb.create_sheet("Third")
    third["A1"] = "third"
    third["C3"] = 3
    third.merge_cells("C3:D4")
    src = tmp_path / "src.xlsx"
    wb.save(src)
    import zipfile

    with zipfile.ZipFile(src) as z:
        s1 = z.read("xl/worksheets/sheet1.xml")
        s3 = z.read("xl/worksheets/sheet3.xml")
        rels = z.read("xl/_rels/workbook.xml.rels").decode()
    # Swap the two files and point the relationships at them with relative targets.
    rels = rels.replace('Target="/xl/worksheets/sheet1.xml"', 'Target="worksheets/sheetX.xml"')
    rels = rels.replace('Target="/xl/worksheets/sheet3.xml"', 'Target="worksheets/sheet1.xml"')
    rels = rels.replace('Target="worksheets/sheetX.xml"', 'Target="./worksheets/sheet3.xml"')
    dst = tmp_path / "swapped.xlsx"
    _patch_xlsx(
        src,
        dst,
        {
            "xl/worksheets/sheet1.xml": s3,
            "xl/worksheets/sheet3.xml": s1,
            "xl/_rels/workbook.xml.rels": rels.encode(),
        },
    )
    g = read_grid(dst, "First")
    assert g.merged == [(1, 1, 1, 2)]
    assert [[(c.raw, c.kind) for c in r] for r in g.rows] == [
        [("first", "text")],
        [("", "formula_no_value")],
    ]
    g = read_grid(dst, "Third")
    assert g.merged == [(3, 3, 4, 4)]
    assert g.cell(1, 1).raw == "third" and g.cell(3, 3).raw == "3"
    assert {c.kind for r in g.rows for c in r} == {"text", "number", "empty"}
    assert read_grid(dst, "Second").rows == [[Cell(1, 1, "second", "text")]]


# ------------------------------------------------- hand-written sheet XML (review findings)

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _sheet_xlsx(tmp_path, sheet_data, *, full_calc=False, name="hand.xlsx"):
    """An xlsx whose one sheet "S" is `<sheetData>{sheet_data}</sheetData>`; `full_calc` keeps
    openpyxl's `fullCalcOnLoad="1"`, otherwise it's dropped (as Excel or LibreOffice save)."""
    import zipfile

    wb = openpyxl.Workbook()
    wb.active.title = "S"
    src = tmp_path / f"src-{name}"
    wb.save(src)
    with zipfile.ZipFile(src) as z:
        workbook = z.read("xl/workbook.xml")
    assert b'fullCalcOnLoad="1"' in workbook
    if not full_calc:
        workbook = _without_full_calc(workbook)
    sheet = f'<worksheet xmlns="{_NS}"><sheetData>{sheet_data}</sheetData></worksheet>'
    dst = tmp_path / name
    _patch_xlsx(src, dst, {"xl/worksheets/sheet1.xml": sheet.encode(), "xl/workbook.xml": workbook})
    return dst


def _cells(g):
    return [[(c.raw, c.kind) for c in r] for r in g.rows]


def test_formula_cached_empty_string_is_empty_not_a_reject(tmp_path):
    """`=IF(..,"",..)` saved by Excel/LibreOffice is `t="str"` with an empty (or self-closed)
    `<v>`: a cached "" that Excel shows as a blank, so `empty`. Without `t="str"`, an empty
    `<v>` (or none at all) is still no cached value (openpyxl writes formulas that way)."""
    path = _sheet_xlsx(
        tmp_path,
        '<row r="1"><c r="A1"><v>1</v></c>'
        '<c r="B1" t="str"><f>IF(A1=1,"",A1)</f><v></v></c>'
        '<c r="C1" t="str"><f>IF(A1=1,"",A1)</f><v/></c>'
        '<c r="D1" t="str"><f>IF(A1=1,"",A1)</f></c>'  # t="str" but no <v>: not calculated
        '<c r="E1"><f>A1*2</f><v></v></c>'  # empty <v> without t="str": not calculated
        '<c r="F1" t="str"><f>"x"&amp;A1</f><v>x1</v></c>'
        '<c r="G1"><v>9</v></c></row>',
    )
    g = read_grid(path)
    fnv = ("", "formula_no_value")
    assert _cells(g) == [
        [("1", "number"), ("", "empty"), ("", "empty"), fnv, fnv, ("x1", "text"), ("9", "number")]
    ]
    assert g.recalc_on_open is False
    assert read_failure_reason(g, g.cell(1, 4)) == FORMULA_NO_VALUE
    assert read_failure_reason(g, g.cell(1, 2)) is None


def test_full_calc_on_load_formulas_are_never_their_placeholder(tmp_path):
    """XlsxWriter (what pandas uses) writes an uncalculated formula as `<f>D2*2</f><v>0</v>` and
    sets `<calcPr fullCalcOnLoad="1"/>`. With that flag no formula's `<v>` is trusted: every
    formula cell, and every cell of an array formula's range, is `formula_no_value`, with its
    own reject reason. Plain values are read as usual."""
    data = (
        '<row r="1"><c r="A1"><v>3</v></c><c r="B1"><f>A1*2</f><v>0</v></c>'
        '<c r="C1" t="str"><f>IF(A1=1,"",A1)</f><v></v></c>'
        '<c r="D1" t="e"><f>1/0</f><v>#DIV/0!</v></c></row>'
        '<row r="2"><c r="A2"><v>4</v></c><c r="B2"><f t="array" ref="B2:B3">A2:A3*2</f>'
        "<v>0</v></c></row>"
        '<row r="3"><c r="A3"><v>5</v></c><c r="B3"><v>0</v></c></row>'
    )
    g = read_grid(_sheet_xlsx(tmp_path, data, full_calc=True))
    fnv = ("", "formula_no_value")
    assert _cells(g) == [
        [("3", "number"), fnv, fnv, fnv],
        [("4", "number"), fnv, ("", "empty"), ("", "empty")],
        [("5", "number"), fnv, ("", "empty"), ("", "empty")],
    ]
    assert g.recalc_on_open is True
    assert read_failure_reason(g, g.cell(1, 2)) == FORMULA_NOT_CALCULATED
    assert "recalculation on open" in FORMULA_NOT_CALCULATED
    assert read_failure_reason(g, g.cell(1, 1)) is None
    with pytest.raises(ParseError):
        run_ops(g.cell(1, 2).raw, [{"op": "number"}])

    # The same sheet saved by Excel or LibreOffice (no flag) reads its cached values.
    g = read_grid(_sheet_xlsx(tmp_path, data, name="saved.xlsx"))
    assert g.recalc_on_open is False
    assert _cells(g)[0] == [("3", "number"), ("0", "number"), ("", "empty"), ("#DIV/0!", "error")]
    assert g.cell(3, 2) == Cell(3, 2, "0", "number")


def test_openpyxl_written_workbook_rejects_with_recalc_reason(plan_xlsx):
    g = read_grid(plan_xlsx, "Plan")
    assert g.recalc_on_open is True  # openpyxl sets fullCalcOnLoad
    assert read_failure_reason(g, g.cell(4, 4)) == FORMULA_NOT_CALCULATED
    assert read_failure_reason(g, g.cell(5, 2)) == "error cell"


def test_committed_formula_fixture_has_real_cached_values():
    """formulas.xlsx was saved by LibreOffice, which doesn't set fullCalcOnLoad: its cached
    values are real, so only the stripped D4 is formula_no_value."""
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "wrangle" / "formulas.xlsx"
    g = read_grid(path, "Calc")
    assert g.recalc_on_open is False
    assert [c.ref for r in g.rows for c in r if c.kind == "formula_no_value"] == ["D4"]


@pytest.mark.parametrize(
    ("sheet_data", "first_bad"),
    [
        # a cell before an earlier column: read-only openpyxl loses C1
        ('<row r="1"><c r="C1"><v>3</v></c><c r="A1"><v>1</v></c></row>', "A1"),
        # rows out of order: row 3 is lost
        ('<row r="5"><c r="A5"><v>5</v></c></row><row r="3"><c r="A3"><v>3</v></c></row>', "A3"),
        # the same row twice: the second is lost
        ('<row r="2"><c r="A2"><v>1</v></c></row><row r="2"><c r="B2"><v>2</v></c></row>', "B2"),
        # a cell off its row: the value lands at A2, the formula pass looks at A3
        ('<row r="2"><c r="A3"><f>1+1</f></c></row>', "A3"),
        # the same cell twice
        ('<row r="1"><c r="A1"><v>1</v></c><c r="A1"><v>2</v></c></row>', "A1"),
        # an empty row out of order
        ('<row r="4"><c r="A4"><v>1</v></c></row><row r="4"/>', "row 4"),
        # a row without `r` is numbered on, so an explicit earlier row after it is out of order
        (
            '<row r="2"><c r="A2"><v>1</v></c></row><row><c><v>2</v></c></row>'
            '<row r="3"><c r="A3"><v>3</v></c></row>',
            "A3",
        ),
    ],
)
def test_out_of_order_sheet_xml_is_refused_not_misread(tmp_path, sheet_data, first_bad):
    path = _sheet_xlsx(tmp_path, sheet_data)
    with pytest.raises(ValueError) as e:
        read_grid(path, "S")
    msg = str(e.value)
    assert str(path) in msg and "'S'" in msg and f"(first at {first_bad})" in msg
    assert "not in row/column order" in msg
    assert "open and re-save the file in Excel or LibreOffice" in msg


def test_rows_and_cells_without_r_still_read(tmp_path):
    """No `r` anywhere, or mixed: numbered on from the previous row or cell, as openpyxl does."""
    path = _sheet_xlsx(
        tmp_path,
        "<row><c><v>1</v></c><c><v>2</v></c></row>"
        '<row><c r="B2"><v>3</v></c><c><v>4</v></c></row>'
        '<row r="4"><c><v>5</v></c><c r="C4"><v>6</v></c><c><f>1+1</f></c></row>',
    )
    assert _cells(read_grid(path)) == [
        [("1", "number"), ("2", "number"), ("", "empty"), ("", "empty")],
        [("", "empty"), ("3", "number"), ("4", "number"), ("", "empty")],
        [("", "empty"), ("", "empty"), ("", "empty"), ("", "empty")],
        [("5", "number"), ("", "empty"), ("6", "number"), ("", "formula_no_value")],
    ]


# --- XlsxBook: one open and one hash for several sheets ---------------------------------------


def test_book_reads_the_same_grids_and_hashes_once(plan_xlsx, monkeypatch):
    """Every sheet (and the default, and a missing one) through one book is exactly what
    `read_grid` alone gives; the file is read for its hash once and opened once."""
    plain = {name: read_grid(plan_xlsx, name) for name in (None, "Plan", "Notes")}
    with pytest.raises(ValueError) as alone:
        read_grid(plan_xlsx, "Forecast")

    reads, loads = [], []
    real_read_bytes, real_load = type(plan_xlsx).read_bytes, openpyxl.load_workbook
    monkeypatch.setattr(
        type(plan_xlsx), "read_bytes", lambda self: reads.append(self) or real_read_bytes(self)
    )
    monkeypatch.setattr(
        openpyxl, "load_workbook", lambda *a, **k: loads.append(a) or real_load(*a, **k)
    )
    with XlsxBook(plan_xlsx) as book:
        assert reads == [] and loads == []  # nothing touched until first use
        for name, grid in plain.items():
            assert read_grid(plan_xlsx, name, book=book) == grid
        assert read_grid(plan_xlsx, "Plan", book=book) == plain["Plan"]  # twice is fine
        with pytest.raises(ValueError) as booked:
            read_grid(plan_xlsx, "Forecast", book=book)
        assert str(booked.value) == str(alone.value)
        assert book.file_sha256 == hashlib.sha256(plan_xlsx.read_bytes()).hexdigest()
    assert len(reads) == 2 and len(loads) == 1  # one hash (+ the line above), one open
    assert book._wb is None and book._zf is None  # closed


def test_book_must_be_the_same_xlsx(plan_xlsx, tmp_path):
    other = tmp_path / "other.xlsx"
    other.write_bytes(plan_xlsx.read_bytes())
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    with XlsxBook(plan_xlsx) as book:
        with pytest.raises(ValueError, match="book= is an XlsxBook of"):
            read_grid(other, "Plan", book=book)
        with pytest.raises(ValueError, match="book= is an XlsxBook of"):
            read_grid(csv_path, book=book)
