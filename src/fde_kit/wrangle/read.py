"""Read a CSV or XLSX sheet into a `Grid`: every cell as a string, with its coordinates.

Everything downstream (profile, spec ops, provenance) works on strings with a `kind`, never on
the Python objects openpyxl returns. That keeps one parse path for CSV and XLSX, and it makes
`source_ref` read-back exact: re-reading the cell gives the same string, which the same ops turn
into the same value.

Two XLSX cases matter because getting them wrong silently loads a wrong number:

- A formula cell with no cached value (a file written by a script and never opened in Excel)
  reads as `None` under `data_only=True`. It's `formula_no_value`, so a spec that reads it
  rejects the cell instead of loading 0 or skipping it as empty.
- Excel error values (`#N/A`, `#REF!`, ...) are `error`, never text that a `map` could rescue.

Numbers are `str(Decimal(repr(float)))` in plain notation (no exponent), so `0.1` is `"0.1"`
and not `0.1000000000000000055...`, and the `number` op's strict grammar accepts it.

`merged` ranges are `(min_row, min_col, max_row, max_col)`, 1-based and inclusive. Only the
top-left cell of a range holds a value; `read` doesn't fill the others (header joining does).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import IO, Literal

CellKind = Literal["empty", "text", "number", "date", "bool", "error", "formula_no_value"]

EXCEL_ERRORS = frozenset(
    {
        "#N/A",
        "#REF!",
        "#DIV/0!",
        "#VALUE!",
        "#NAME?",
        "#NUM!",
        "#NULL!",
        "#SPILL!",
        "#CALC!",
        "#GETTING_DATA",
    }
)
XLSX_SUFFIXES = frozenset({".xlsx", ".xlsm"})
CSV_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})
SNIFF_BYTES = 64 * 1024
SNIFF_DELIMITERS = ",;\t|"

_REF_RE = re.compile(r"\$?([A-Za-z]{1,3})\$?([1-9][0-9]*)")


def col_letter(n: int) -> str:
    """1 -> "A", 26 -> "Z", 27 -> "AA"."""
    if n < 1:
        raise ValueError(f"column numbers are 1-based, got {n}")
    letters = ""
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def col_number(letters: str) -> int:
    """ "A" -> 1, "AA" -> 27. The inverse of `col_letter`."""
    if not letters.isalpha() or not letters.isascii():
        raise ValueError(f"not a column: {letters!r}")
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def parse_ref(ref: str) -> tuple[int, int]:
    """ "G42" -> (42, 7), as (row, col). `$` anchors and lower case are accepted."""
    m = _REF_RE.fullmatch(ref.strip())
    if not m:
        raise ValueError(f"not an A1 cell reference: {ref!r}")
    return int(m.group(2)), col_number(m.group(1))


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    """1-based, as Excel shows them."""
    raw: str
    """Text as stored; numbers as plain-notation str(Decimal); dates ISO."""
    kind: CellKind

    @property
    def ref(self) -> str:
        return f"{col_letter(self.col)}{self.row}"


@dataclass(frozen=True)
class Grid:
    path: str
    sheet: str
    """The XLSX sheet name; for CSV, the file stem."""
    file_sha256: str
    rows: list[list[Cell]]
    """Rectangular: every row has the same length, padded with empty cells."""
    merged: list[tuple[int, int, int, int]]
    """(min_row, min_col, max_row, max_col), 1-based, inclusive; empty for CSV."""

    def cell(self, row: int, col: int) -> Cell:
        """The cell at 1-based (row, col); an empty cell outside the read area."""
        if 1 <= row <= len(self.rows) and 1 <= col <= len(self.rows[row - 1]):
            return self.rows[row - 1][col - 1]
        return Cell(row, col, "", "empty")


def read_grid(
    path: str | Path,
    sheet: str | None = None,
    *,
    encoding: str = "utf-8-sig",
    delimiter: str | None = None,
) -> Grid:
    """Read one sheet. XLSX: `sheet` defaults to the first one. CSV: `sheet` must be omitted or
    equal the file stem; `encoding` and `delimiter` apply to CSV only."""
    p = Path(path)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    suffix = p.suffix.lower()
    if suffix in XLSX_SUFFIXES:
        return _read_xlsx(p, sheet, sha)
    if suffix in CSV_SUFFIXES:
        if sheet is not None and sheet != p.stem:
            raise ValueError(f"{p.name} is a CSV; its only sheet is {p.stem!r}, not {sheet!r}")
        return _read_csv(p, data, sha, encoding, delimiter)
    raise ValueError(f"unsupported file type {p.suffix!r} (want .csv or .xlsx): {p}")


def _trim(rows: list[list[Cell]]) -> list[list[Cell]]:
    """Drop trailing rows and columns that are entirely empty, then pad to a rectangle."""
    while rows and all(c.kind == "empty" for c in rows[-1]):
        rows.pop()
    width = 0
    for r in rows:
        for c in reversed(r):
            if c.kind != "empty":
                width = max(width, c.col)
                break
    out = []
    for i, r in enumerate(rows, start=1):
        r = r[:width]
        r.extend(Cell(i, j, "", "empty") for j in range(len(r) + 1, width + 1))
        out.append(r)
    return out


def _plain_decimal(d: Decimal) -> str:
    s = str(d)
    return format(d, "f") if "E" in s or "e" in s else s


def _xlsx_cell(row: int, col: int, value: object, is_formula: bool) -> Cell:
    if value is None:
        return Cell(row, col, "", "formula_no_value" if is_formula else "empty")
    if isinstance(value, bool):
        return Cell(row, col, "true" if value else "false", "bool")
    if isinstance(value, int):
        return Cell(row, col, str(value), "number")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return Cell(row, col, repr(value), "error")
        return Cell(row, col, _plain_decimal(Decimal(repr(value))), "number")
    if isinstance(value, Decimal):
        return Cell(row, col, _plain_decimal(value), "number")
    if isinstance(value, dt.datetime):
        if value.time() == dt.time(0, 0):
            return Cell(row, col, value.date().isoformat(), "date")
        return Cell(row, col, value.isoformat(), "date")
    if isinstance(value, dt.date):
        return Cell(row, col, value.isoformat(), "date")
    if isinstance(value, dt.time):
        return Cell(row, col, value.isoformat(), "text")
    text = str(value)
    if text in EXCEL_ERRORS:
        return Cell(row, col, text, "error")
    return Cell(row, col, text, "empty" if not text.strip() else "text")


_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PKG_RELS_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_DOC_RELS_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)


def _rels_path(part: str) -> str:
    """ "xl/workbook.xml" -> "xl/_rels/workbook.xml.rels"."""
    folder, name = posixpath.split(part)
    return posixpath.join(folder, "_rels", name + ".rels")


def _rel_targets(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Relationship id -> (type, zip member path) for one part's `.rels`; {} if it has none."""
    try:
        data = zf.read(_rels_path(part))
    except KeyError:
        return {}
    out = {}
    base = posixpath.dirname(part)
    for rel in ET.fromstring(data).iter(f"{_PKG_RELS_NS}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target", "")
        # A target starting with "/" is relative to the package root, otherwise to the part.
        joined = target.lstrip("/") if target.startswith("/") else posixpath.join(base, target)
        out[rel.get("Id", "")] = (rel.get("Type", ""), posixpath.normpath(joined))
    return out


def _sheet_xml_path(zf: zipfile.ZipFile, sheet: str) -> str:
    """The zip member holding `sheet`'s XML: sheet name -> r:id in the workbook part ->
    relationship target. File names (`sheet3.xml`) need not follow sheet order."""
    workbook = next(
        (t for typ, t in _rel_targets(zf, "").values() if typ == _OFFICE_DOCUMENT_REL),
        "xl/workbook.xml",
    )
    rels = _rel_targets(zf, workbook)
    for el in ET.fromstring(zf.read(workbook)).iter(f"{_MAIN_NS}sheet"):
        if el.get("name") == sheet:
            rid = el.get(f"{_DOC_RELS_NS}id")
            if rid in rels:
                return rels[rid][1]
            break
    raise ValueError(f"no worksheet part found for sheet {sheet!r}")


def _row_number(r: str) -> int:
    """A `<row r="..">`, read as openpyxl does ("3", or a whole float like "3.0")."""
    try:
        return int(r)
    except ValueError:
        val = float(r)
        if not val.is_integer():
            raise ValueError(f"{r} is not a valid row number") from None
        return int(val)


def _scan_sheet_xml(
    src: IO[bytes],
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """One streaming pass over a sheet's XML for what read-only openpyxl doesn't expose: the
    cells that hold a formula but no cached value (`<f>`, shared formulas' empty `<f t="shared"
    si=".."/>` included, with no or an empty `<v>`), and the `<mergeCells>` ranges.

    Cell coordinates follow openpyxl's parser exactly, including rows and cells written without
    an `r` attribute (numbered on from the previous one), so both passes agree on every cell.
    """
    from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

    no_value: set[tuple[int, int]] = set()
    merged: list[tuple[int, int, int, int]] = []
    c_tag, row_tag, f_tag, v_tag = (f"{_MAIN_NS}{t}" for t in ("c", "row", "f", "v"))
    merge_tag, data_tag = f"{_MAIN_NS}mergeCell", f"{_MAIN_NS}sheetData"
    sheet_data = None
    row = 0
    for event, el in ET.iterparse(src, events=("start", "end")):
        if event == "start":
            if el.tag == data_tag:
                sheet_data = el
            continue
        tag = el.tag
        if tag == row_tag:
            # A row's cells are read when the row ends; then the row is dropped.
            r = el.get("r")
            row = row + 1 if r is None else _row_number(r)
            col = 0  # the previous cell's column, as openpyxl counts it
            last_ref = None  # ... or the previous cell's `r`, parsed only when needed
            for c in el:
                if c.tag != c_tag:
                    continue
                ref = c.get("r")
                if ref:
                    last_ref = ref
                else:
                    if last_ref is not None:
                        col = coordinate_to_tuple(last_ref)[1]
                        last_ref = None
                    col += 1
                if c.find(f_tag) is not None and not c.findtext(v_tag):
                    no_value.add(coordinate_to_tuple(ref) if ref else (row, col))
            el.clear()
            if sheet_data is not None:
                sheet_data.clear()
        elif tag == merge_tag:
            ref = el.get("ref")
            if ref:
                min_col, min_row, max_col, max_row = range_boundaries(ref)
                merged.append((min_row, min_col, max_row, max_col))
    return no_value, merged


def _read_xlsx(p: Path, sheet: str | None, sha: str) -> Grid:
    """Two streaming passes over the requested sheet only, both on public or standard APIs.

    1. Values: `openpyxl.load_workbook(read_only=True, data_only=True)` and `iter_rows` on the
       one sheet, so values are exactly what openpyxl's `data_only` load gives (dates from the
       number formats, shared strings, errors as text). The declared `<dimension>` is ignored
       (`reset_dimensions`): writers get it wrong, and it would truncate or pad the rows.
    2. A stdlib `iterparse` of the same sheet XML (found through the workbook's relationships)
       for formula cells without a cached value and the `<mergeCells>`, which read-only
       worksheets don't expose. A formula with no cached value reads as `None` in pass 1, the
       same as an empty cell; pass 2 is what tells them apart.

    Memory is the grid itself, not openpyxl's worksheet model.
    """
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("reading XLSX needs openpyxl: pip install 'fde-kit[wrangle]'") from e

    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    try:
        names = wb.sheetnames
        name = sheet if sheet is not None else names[0]
        if name not in names:
            raise ValueError(f"{p.name} has no sheet {name!r}; available sheets: {names}")
        ws = wb[name]
        if ws not in wb.worksheets:  # a chartsheet
            raise ValueError(f"{p.name}: sheet {name!r} is not a worksheet")
        ws.reset_dimensions()
        values: dict[tuple[int, int], object] = {}
        for r_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            for c_idx, value in enumerate(row, start=1):
                if value is not None:
                    values[(r_idx, c_idx)] = value
    finally:
        wb.close()

    with zipfile.ZipFile(p) as zf, zf.open(_sheet_xml_path(zf, name)) as src:
        no_value, merged = _scan_sheet_xml(src)

    sparse = {k: _xlsx_cell(k[0], k[1], v, False) for k, v in values.items()}
    for r_idx, c_idx in no_value:
        if (r_idx, c_idx) not in values:
            sparse[(r_idx, c_idx)] = Cell(r_idx, c_idx, "", "formula_no_value")

    for min_row, min_col, max_row, max_col in merged:
        # As openpyxl's full model does: only the top-left cell of a merged range keeps a value.
        if (max_row - min_row + 1) * (max_col - min_col + 1) <= len(sparse):
            for r_idx in range(min_row, max_row + 1):
                for c_idx in range(min_col, max_col + 1):
                    if (r_idx, c_idx) != (min_row, min_col):
                        sparse.pop((r_idx, c_idx), None)
        else:
            for key in [k for k in sparse if min_row <= k[0] <= max_row]:
                if min_col <= key[1] <= max_col and key != (min_row, min_col):
                    del sparse[key]
    merged.sort()

    # Size from the non-empty cells only, so styled-but-empty cells far out don't widen the grid.
    height = width = 0
    for (r_idx, c_idx), cell in sparse.items():
        if cell.kind != "empty":
            height = max(height, r_idx)
            width = max(width, c_idx)
    rows = [
        [
            sparse.get((r_idx, c_idx)) or Cell(r_idx, c_idx, "", "empty")
            for c_idx in range(1, width + 1)
        ]
        for r_idx in range(1, height + 1)
    ]
    return Grid(path=str(p), sheet=name, file_sha256=sha, rows=_trim(rows), merged=merged)


def sniff_delimiter(sample: str) -> str:
    """The delimiter csv.Sniffer picks from `,;\\t|`, or "," when it can't tell."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=SNIFF_DELIMITERS).delimiter
    except csv.Error:
        return ","


def _read_csv(p: Path, data: bytes, sha: str, encoding: str, delimiter: str | None) -> Grid:
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as e:
        bad = data[e.start : e.start + 1].hex()
        raise ValueError(
            f"{p}: not valid {encoding} at byte offset {e.start} (byte 0x{bad}); "
            "set `source.encoding` in the spec to the file's encoding (e.g. latin-1 or cp1252)"
        ) from e
    except LookupError as e:
        raise ValueError(f"{p}: unknown encoding {encoding!r}") from e
    if delimiter is None:
        delimiter = sniff_delimiter(text[:SNIFF_BYTES])
    rows: list[list[Cell]] = []
    for r_idx, record in enumerate(
        csv.reader(io.StringIO(text, newline=""), delimiter=delimiter), start=1
    ):
        rows.append(
            [
                Cell(r_idx, c_idx, v, "empty" if not v.strip() else "text")
                for c_idx, v in enumerate(record, start=1)
            ]
        )
    return Grid(path=str(p), sheet=p.stem, file_sha256=sha, rows=_trim(rows), merged=[])
