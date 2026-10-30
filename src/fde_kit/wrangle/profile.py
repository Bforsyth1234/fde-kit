"""Profile a sheet deterministically (§7.2 of docs/pr5-wrangle-core.md).

`profile(grid)` says where the table is and what is in it: header and title rows, the data
region, totals rows to skip, per-column types and formats, scale hints, wide (period-across)
layouts and key candidates. It is what a person reads before writing a mapping spec, and what
the PR 7 agent will read. The profile proposes; the spec decides; apply never reads it.

Same grid in, same bytes out: `Profile.model_dump_json()` has a fixed field order, rounded
rates and no paths or times, and `profile_sha256` is the sha256 of that JSON with
`profile_sha256` itself left out.

Heuristics, including the choices §7.2 leaves open:

- **Text vs numeric.** CSV cells are all `text`, so "text" means a `text` cell that doesn't
  parse as a number in any of the styles in `NUMBER_STYLES` (with parentheses, trailing minus,
  currency and % allowed). `number`, `date` and `bool` cells, and text that parses as a number,
  are non-text. A date written as text (`30/04/2025`) is text for the header rule. Because
  `parse_number` strips a leading ISO currency code, a letter run is accepted only as one
  3-letter code set off by whitespace (`EUR 1.234`), so `DET-07` is text, not -7.
- **Header.** The first row `r` in rows 1..30 where >= 60% of non-empty cells are text and each
  non-empty row among `r+1..r+5` is >= 60% non-text (at least one such row must exist; empty
  rows, and rows whose first cell is a total label, are ignored: `Total | | | 1540` is 50%
  text and would otherwise hide a header above a short table). If row `r-1` holds a merged
  range at least two columns wide that ends above `r` and covers a column with a label in row
  `r`, the header is `[r-1, r]`. When no row passes, the first non-empty row is used and a
  warning says so.
- **Labels** come from `labels.header_labels` (§7.6), shared with apply.
- **Data region.** From the row after the header to the last non-empty row before 3 consecutive
  empty rows (or the end of the sheet). Every non-empty row after that is a notes row.
- **Skip candidates.** A row whose first non-empty cell matches `(?i)^(sub|grand )?total`
  (reason `label`), or a row where >= 2 numeric-column cells each equal (within 0.5) the sum of
  that column over a window of >= 2 earlier rows (reason `sum`). Two windows are tried: the
  rows since the previous candidate (a subtotal), and all non-candidate rows since the start of
  the region (a grand total that follows the last subtotal). A one-row window is not used: a row
  that repeats the row before it is a duplicate, not a subtotal. Numeric columns here are the
  ones whose cells are >= 95% numeric over the whole region.
- **Column stats** (`null_rate`, `distinct`, `unique`, `top`, `samples`, type, formats) are over
  the data region's non-empty rows minus skip candidates, so a `Subtotal` row doesn't spoil a
  key. Values are compared verbatim (a trailing space makes a different value: messes show).
  `samples` are 10 cells spread evenly down the column. `unique` means no nulls, no error
  cells and no repeats.
- **Type.** `date` if any date format parses >= 95% of the values; else `number` if >= 95%
  are numeric; `bool` if all are bool cells; `text` if none is numeric; otherwise `mixed`.
  Error and formula-without-value cells are counted in `errors` and left out of the type.
- **Number format** (number columns). Each style in `NUMBER_STYLES` is scored by how many text
  values it parses (`number` cells fit every style). The best style wins; a tie goes to the
  style most columns of this sheet chose without a tie, then to the first in the list, and is
  marked `ambiguous`. `thousands`/`decimal` are the style (what to write in the `number` op);
  the `*_seen` flags say what the values actually contain.
- **Date formats.** Each of `DATE_FORMATS` that parses >= 95% of values, best first (ties in
  list order). `day_month_ambiguous` when a day-first and month-first pair both parse every
  value.
- **Scale hint.** From the column's label, else (number columns only) from a title row:
  `(000s)`, `(000)`, `'000`, `in thousands` -> 1000; `in millions`, `(millions)`, `$m`, `$mm`,
  `EUR m`-style currency-symbol `m` -> 1e6. A bare `(m)` is not a hint (it might be metres).
- **Unpivot.** For each column, the bottom part of its joined label (after the last `" / "`)
  is tried against `PERIOD_FORMATS` then `PERIOD_PATTERNS`. A run of >= 3 adjacent columns that
  share one format and the same upper prefix is a candidate; the longest run from each start
  wins, the first format on ties. `from`/`to` are full joined labels, and the suggested `ops` are
  a regex stripping the prefix (when there is one) and the period op. An ISO-date run whose days
  are all 1st or month-end in distinct months is grain `month`, otherwise `day`.
- **Keys.** Text-ish columns are those typed `text`, `date`, `bool` or `mixed`, and `number`
  columns whose values are all bare digit strings of one width >= 3 (codes like `1007`; a
  quantity column has varying widths). Single keys: text-ish
  columns that are `unique` (values stripped). Pairs: among the first 10 text-ish columns, pairs
  with no nulls that are unique together and contain no single key. Keys are named by label
  (the column letter if the label is empty).
"""

from __future__ import annotations

import calendar
import hashlib
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fde_kit.wrangle.labels import LABEL_JOIN, header_fingerprint, header_labels
from fde_kit.wrangle.parse import ParseError, compile_period_pattern, parse_date, parse_number
from fde_kit.wrangle.read import Cell, Grid, col_letter

HEADER_WINDOW = 30
HEADER_TEXT_SHARE = 0.6
HEADER_LOOKAHEAD = 5
EMPTY_ROWS_END = 3
SHARE = 0.95
SUM_TOLERANCE = Decimal("0.5")
TOP_N = 10
SAMPLES_N = 10
KEY_COLUMNS = 10
UNPIVOT_MIN = 3

# (thousands, decimal), in tie-break order.
NUMBER_STYLES: list[tuple[str, str]] = [(",", "."), (".", ","), ("'", "."), (" ", ",")]
_THOUSANDS_SEEN = {",": ",", ".": ".", "'": "'’", " ": "   "}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
    "%Y%m%d",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%b %d, %Y",
]
DAY_MONTH_PAIRS = [("%d/%m/%Y", "%m/%d/%Y"), ("%d/%m/%y", "%m/%d/%y"), ("%d-%m-%Y", "%m-%d-%Y")]

# strptime formats a header label may be a period in; grain None = decided from the dates.
PERIOD_FORMATS: list[tuple[str, str | None]] = [
    ("%b-%y", "month"),
    ("%b %y", "month"),
    ("%b-%Y", "month"),
    ("%b %Y", "month"),
    ("%B %Y", "month"),
    ("%B-%Y", "month"),
    ("%Y-%m", "month"),
    ("%m/%Y", "month"),
    ("%Y-%m-%d", None),
]
PERIOD_PATTERNS: list[tuple[str, str]] = [
    ("Q{q} FY{yy}", "quarter"),
    ("Q{q} FY{yyyy}", "quarter"),
    ("Q{q} {yyyy}", "quarter"),
    ("Q{q}-{yy}", "quarter"),
    ("FY{yy}", "year"),
    ("FY{yyyy}", "year"),
    ("{yyyy}", "year"),
]

TOTAL_LABEL = re.compile(r"(?i)^(sub|grand )?total")
_SCALE_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"(?i)\(\s*0{3}s?\s*\)|'0{3}s?\b|\bin\s+thousands\b|\(\s*thousands\s*\)"), 1000),
    (
        re.compile(
            r"(?i)\bin\s+millions\b|\(\s*millions\s*\)|[$€£]\s?mm?\b|\b(?:usd|eur|gbp)\s?mm?\b"
        ),
        1_000_000,
    ),
]
_BARE_DIGITS = re.compile(r"\d+")
_LETTERS = re.compile(r"[^\W\d_]+")

ColumnType = Literal["empty", "text", "number", "date", "bool", "mixed"]


# ------------------------------------------------------------------------------- models


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CellValue(_Model):
    ref: str
    value: str


class TextRow(_Model):
    row: int
    cells: list[CellValue]
    """The row's non-empty cells, verbatim, left to right."""


class DataRegion(_Model):
    start: int
    end: int
    """Inclusive; `end < start` when there are no data rows."""
    rows: int
    """Non-empty rows in the region."""


class SkipCandidate(_Model):
    row: int
    reasons: list[Literal["label", "sum"]]
    first_value: str
    sum_columns: list[str]
    """Labels of the columns whose cell equals the window sum (reason `sum` only)."""


class NumberFormat(_Model):
    thousands: str
    decimal: str
    ambiguous: bool
    """True when more than one style parsed equally well and the tie was broken by the sheet's
    other columns or by list order."""
    parse_rate: float
    thousands_seen: bool
    decimal_seen: bool
    parens_negative: bool
    trailing_minus: bool
    currency: bool
    percent: bool


class ScaleHint(_Model):
    scale: int
    source: Literal["label", "title_row"]
    ref: str | None
    """The title cell the hint came from; None for a label."""
    text: str


class TopValue(_Model):
    value: str
    count: int


class ColumnProfile(_Model):
    col: int
    letter: str
    label: str
    type: ColumnType
    non_null: int
    null_rate: float
    errors: int
    distinct: int
    unique: bool
    top: list[TopValue]
    samples: list[CellValue]
    number_format: NumberFormat | None
    date_formats: list[str]
    day_month_ambiguous: bool
    scale_hint: ScaleHint | None


class UnpivotCandidate(_Model):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True, validate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    format: str | None
    pattern: str | None
    grain: str
    columns: list[str]
    """Column letters, in sheet order."""
    ops: list[dict[str, Any]]
    """Suggested `period.unpivot.ops` for the spec."""


class Profile(_Model):
    file: str
    """The file's base name (no directories, so the profile is the same wherever it's read)."""
    sheet: str
    file_sha256: str
    n_rows: int
    n_cols: int
    header_rows: list[int]
    header_found: bool
    title_rows: list[TextRow]
    data: DataRegion
    notes_rows: list[TextRow]
    skip_candidates: list[SkipCandidate]
    columns: list[ColumnProfile]
    scale_hints: list[ScaleHint]
    """Every hint found in the title rows."""
    unpivot_candidates: list[UnpivotCandidate]
    keys: list[list[str]]
    header_fingerprint: str
    warnings: list[str]
    profile_sha256: str = ""


# ----------------------------------------------------------------------------- helpers


def _nonempty(row: list[Cell]) -> list[Cell]:
    return [c for c in row if c.kind != "empty"]


def _parse_style(raw: str, style: tuple[str, str]) -> Decimal | None:
    if not _plausible_number(raw):
        return None
    try:
        return parse_number(
            raw,
            thousands=style[0],
            decimal=style[1],
            parens_negative=True,
            trailing_minus=True,
            currency=True,
            percent=True,
        )
    except ParseError:
        return None


def _plausible_number(raw: str) -> bool:
    """`parse_number` strips a leading ISO currency code, so `DET-07` would read as -7. Here a
    letter run is allowed only as one 3-letter code set off by whitespace (`EUR 1.234`)."""
    runs = list(_LETTERS.finditer(raw))
    if not runs:
        return True
    if len(runs) > 1 or not re.fullmatch(r"[A-Z]{3}", runs[0].group()):
        return False
    m = runs[0]
    before, after = raw[: m.start()], raw[m.end() :]
    return after[:1].isspace() or before[-1:].isspace() or (not after.strip(" )"))


def _is_numeric(cell: Cell) -> bool:
    if cell.kind == "number":
        return True
    if cell.kind != "text":
        return False
    return _plausible_number(cell.raw) and any(
        _parse_style(cell.raw, s) is not None for s in NUMBER_STYLES
    )


def _is_text(cell: Cell) -> bool:
    return cell.kind == "text" and not _is_numeric(cell)


def _cell_value(cell: Cell, style: tuple[str, str]) -> Decimal | None:
    if cell.kind == "number":
        try:
            return Decimal(cell.raw)
        except InvalidOperation:
            return None
    if cell.kind == "text":
        return _parse_style(cell.raw, style)
    return None


def _text_row(row: list[Cell]) -> TextRow:
    cells = _nonempty(row)
    return TextRow(row=cells[0].row, cells=[CellValue(ref=c.ref, value=c.raw) for c in cells])


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


# ------------------------------------------------------------------------------ header


def _text_share(cells: list[Cell]) -> float:
    return sum(_is_text(c) for c in cells) / len(cells)


def _find_header(grid: Grid) -> tuple[list[int], bool]:
    n = len(grid.rows)
    for r in range(1, min(HEADER_WINDOW, n) + 1):
        cells = _nonempty(grid.rows[r - 1])
        if not cells or _text_share(cells) < HEADER_TEXT_SHARE:
            continue
        below = [
            _nonempty(grid.rows[i - 1]) for i in range(r + 1, min(r + HEADER_LOOKAHEAD, n) + 1)
        ]
        below = [b for b in below if b and not TOTAL_LABEL.match(b[0].raw.strip())]
        if not below or any(1 - _text_share(b) < HEADER_TEXT_SHARE for b in below):
            continue
        if r > 1 and _upper_header(grid, r):
            return [r - 1, r], True
        return [r], True
    for r, row in enumerate(grid.rows, start=1):
        if _nonempty(row):
            return [r], False
    return [], False


def _upper_header(grid: Grid, r: int) -> bool:
    labelled = {c.col for c in _nonempty(grid.rows[r - 1])}
    for min_row, min_col, max_row, max_col in grid.merged:
        if (
            min_row <= r - 1 <= max_row < r
            and max_col > min_col
            and any(min_col <= c <= max_col for c in labelled)
        ):
            return True
    return False


# --------------------------------------------------------------------------- formats


def _style_scores(cells: list[Cell]) -> list[int]:
    scores = []
    for style in NUMBER_STYLES:
        scores.append(
            sum(
                c.kind == "number" or (c.kind == "text" and _parse_style(c.raw, style) is not None)
                for c in cells
            )
        )
    return scores


def _tied_styles(cells: list[Cell]) -> list[int]:
    """Indices of the best-scoring styles; empty when nothing parses."""
    scores = _style_scores(cells)
    best = max(scores, default=0)
    return [i for i, s in enumerate(scores) if s == best] if best else []


def _pick_style(tied: list[int], preferred: int | None) -> tuple[int, bool]:
    if len(tied) == 1:
        return tied[0], False
    if preferred is not None and preferred in tied:
        return preferred, True
    return tied[0], True


def _sheet_preference(per_column: list[list[int]]) -> int | None:
    votes = Counter(t[0] for t in per_column if len(t) == 1)
    if not votes:
        return None
    return min(votes, key=lambda i: (-votes[i], i))


def _number_format(cells: list[Cell], idx: int, ambiguous: bool) -> NumberFormat:
    style = NUMBER_STYLES[idx]
    parsed = [c for c in cells if c.kind == "number" or _parse_style(c.raw, style) is not None]
    texts = [c.raw.strip() for c in parsed if c.kind == "text"]
    return NumberFormat(
        thousands=style[0],
        decimal=style[1],
        ambiguous=ambiguous,
        parse_rate=_rate(len(parsed), len(cells)),
        thousands_seen=any(ch in s for s in texts for ch in _THOUSANDS_SEEN[style[0]]),
        decimal_seen=any(style[1] in s for s in texts),
        parens_negative=any(s.startswith("(") and s.endswith(")") for s in texts),
        trailing_minus=any(len(s) > 1 and s.endswith("-") for s in texts),
        currency=any(re.search(r"[$€£¥]|[A-Z]{3}", s) is not None for s in texts),
        percent=any(s.endswith("%") for s in texts),
    )


def _date_formats(raws: list[str]) -> tuple[list[str], bool]:
    n = len(raws)
    if not n:
        return [], False
    allowed = int(n * (1 - SHARE) + 1e-9)  # failures tolerated while staying >= 95%
    ok: dict[str, int] = {}
    for fmt in DATE_FORMATS:
        fails = 0
        for raw in raws:
            try:
                parse_date(raw, format=fmt)
            except ParseError:
                fails += 1
                if fails > allowed:
                    break
        if fails <= allowed:
            ok[fmt] = n - fails
    ranked = sorted(ok, key=lambda f: (-ok[f], DATE_FORMATS.index(f)))
    ambiguous = any(ok.get(a) == n and ok.get(b) == n for a, b in DAY_MONTH_PAIRS)
    return ranked, ambiguous


def _scale_of(text: str) -> int | None:
    for rx, scale in _SCALE_RULES:
        if rx.search(text):
            return scale
    return None


# --------------------------------------------------------------------------- regions


def _data_region(grid: Grid, start: int) -> tuple[int, list[int]]:
    """(end, notes row numbers)."""
    n = len(grid.rows)
    end = start - 1
    empties = 0
    r = start
    while r <= n:
        if _nonempty(grid.rows[r - 1]):
            end = r
            empties = 0
        else:
            empties += 1
            if empties >= EMPTY_ROWS_END:
                break
        r += 1
    notes = [i for i in range(end + 1, n + 1) if _nonempty(grid.rows[i - 1])]
    return end, notes


def _skip_candidates(
    grid: Grid, rows: list[int], numeric_cols: dict[int, tuple[str, str]], labels: list[str]
) -> list[SkipCandidate]:
    out: list[SkipCandidate] = []
    window: list[int] = []
    kept: list[int] = []

    def value(r: int, c: int) -> Decimal | None:
        return _cell_value(grid.cell(r, c), numeric_cols[c])

    def sum_matches(r: int, span: list[int]) -> list[int]:
        if len(span) < 2:
            return []
        hits = []
        for c in numeric_cols:
            v = value(r, c)
            if v is None:
                continue
            vals = [x for x in (value(i, c) for i in span) if x is not None]
            if vals and abs(sum(vals, Decimal(0)) - v) <= SUM_TOLERANCE:
                hits.append(c)
        return hits

    for r in rows:
        cells = _nonempty(grid.rows[r - 1])
        first = cells[0].raw.strip()
        reasons: list[Literal["label", "sum"]] = []
        if TOTAL_LABEL.match(first):
            reasons.append("label")
        hits = sum_matches(r, window)
        if len(hits) < 2:
            hits = sum_matches(r, kept)
        if len(hits) >= 2:
            reasons.append("sum")
        else:
            hits = []
        if reasons:
            out.append(
                SkipCandidate(
                    row=r,
                    reasons=reasons,
                    first_value=cells[0].raw,
                    sum_columns=[labels[c - 1] or col_letter(c) for c in hits],
                )
            )
            window = []
        else:
            window.append(r)
            kept.append(r)
    return out


# --------------------------------------------------------------------------- columns


def _column_type(cells: list[Cell], date_formats: list[str]) -> tuple[ColumnType, int]:
    if not cells:
        return "empty", 0
    numeric = sum(_is_numeric(c) for c in cells)
    if date_formats:
        return "date", numeric
    if numeric >= SHARE * len(cells):
        return "number", numeric
    if all(c.kind == "bool" for c in cells):
        return "bool", numeric
    if numeric == 0:
        return "text", numeric
    return "mixed", numeric


def _spread(items: list[Cell], n: int) -> list[Cell]:
    if len(items) <= n:
        return items
    return [items[i * len(items) // n] for i in range(n)]


def _texty(col: ColumnProfile, cells: list[Cell]) -> bool:
    if col.type in ("text", "date", "bool", "mixed"):
        return True
    if col.type != "number" or not cells:
        return False
    raws = [c.raw.strip() for c in cells]
    widths = {len(r) for r in raws}
    return all(_BARE_DIGITS.fullmatch(r) for r in raws) and len(widths) == 1 and min(widths) >= 3


# --------------------------------------------------------------------------- unpivot


def _period_matcher(spec: tuple[str, str | None], is_pattern: bool):
    if is_pattern:
        rx = compile_period_pattern(spec[0], spec[1] or "year")
        return lambda s: rx.fullmatch(s.strip()) is not None
    fmt = spec[0]

    def match(s: str) -> bool:
        try:
            parse_date(s, format=fmt)
        except ParseError:
            return False
        return True

    return match


def _iso_grain(parts: list[str]) -> str:
    dates = [parse_date(p, format="%Y-%m-%d") for p in parts]
    months = {(d.year, d.month) for d in dates}
    edges = all(d.day in (1, calendar.monthrange(d.year, d.month)[1]) for d in dates)
    return "month" if edges and len(months) == len(dates) else "day"


def _split_label(label: str) -> tuple[str, str]:
    if LABEL_JOIN in label:
        prefix, bottom = label.rsplit(LABEL_JOIN, 1)
        return prefix, bottom
    return "", label


def _regex_literal(s: str) -> str:
    """re.escape, but leaving spaces alone so the suggested pattern stays readable."""
    return re.escape(s).replace("\\ ", " ")


def _unpivot_candidates(labels: list[str]) -> list[UnpivotCandidate]:
    split = [_split_label(lab) for lab in labels]
    matchers = [(f, False, _period_matcher(f, False)) for f in PERIOD_FORMATS] + [
        (p, True, _period_matcher(p, True)) for p in PERIOD_PATTERNS
    ]
    out: list[UnpivotCandidate] = []
    i, n = 0, len(labels)
    while i < n:
        best_len, best = 0, None
        if split[i][1]:
            for spec, is_pattern, match in matchers:
                j = i
                while j < n and split[j][1] and split[j][0] == split[i][0] and match(split[j][1]):
                    j += 1
                if j - i > best_len:
                    best_len, best = j - i, (spec, is_pattern)
        if best is None or best_len < UNPIVOT_MIN:
            i += 1
            continue
        (text, grain), is_pattern = best
        bottoms = [split[k][1].strip() for k in range(i, i + best_len)]
        if grain is None:
            grain = _iso_grain(bottoms)
        prefix = split[i][0]
        ops: list[dict[str, Any]] = []
        if prefix:
            ops.append({"op": "regex", "pattern": _regex_literal(prefix + LABEL_JOIN) + "(.+)"})
        period: dict[str, Any] = {"op": "period"}
        period["pattern" if is_pattern else "format"] = text
        period["grain"] = grain
        ops.append(period)
        out.append(
            UnpivotCandidate(
                from_=labels[i],
                to=labels[i + best_len - 1],
                format=None if is_pattern else text,
                pattern=text if is_pattern else None,
                grain=grain,
                columns=[col_letter(k + 1) for k in range(i, i + best_len)],
                ops=ops,
            )
        )
        i += best_len
    return out


# ----------------------------------------------------------------------------- profile


def profile(grid: Grid) -> Profile:
    """Profile one sheet. Deterministic: the same grid gives byte-identical JSON."""
    width = len(grid.rows[0]) if grid.rows else 0
    warnings: list[str] = []

    header_rows, found = _find_header(grid)
    if not found and header_rows:
        warnings.append(
            f"no row in the first {HEADER_WINDOW} passed the header rule; "
            f"using the first non-empty row, {header_rows[0]}"
        )
    labels = header_labels(grid, header_rows)
    first_header = header_rows[0] if header_rows else 1
    title_rows = [
        _text_row(grid.rows[r - 1]) for r in range(1, first_header) if _nonempty(grid.rows[r - 1])
    ]

    start = header_rows[-1] + 1 if header_rows else 1
    end, notes = _data_region(grid, start)
    data_rows = [r for r in range(start, end + 1) if _nonempty(grid.rows[r - 1])]

    # Pass 1, over every data row: which columns are numeric, and in what style.
    all_cells = {
        c: [grid.cell(r, c) for r in data_rows if grid.cell(r, c).kind != "empty"]
        for c in range(1, width + 1)
    }
    tied1 = {c: _tied_styles(cells) for c, cells in all_cells.items()}
    pref1 = _sheet_preference(list(tied1.values()))
    numeric_cols = {
        c: NUMBER_STYLES[_pick_style(tied1[c], pref1)[0]]
        for c, cells in all_cells.items()
        if cells and tied1[c] and sum(_is_numeric(x) for x in cells) >= SHARE * len(cells)
    }
    skips = _skip_candidates(grid, data_rows, numeric_cols, labels)
    skipped = {s.row for s in skips}
    kept = [r for r in data_rows if r not in skipped]

    # Title-row scale hints.
    title_hints = []
    for tr in title_rows:
        for cv in tr.cells:
            scale = _scale_of(cv.value)
            if scale is not None:
                title_hints.append(
                    ScaleHint(scale=scale, source="title_row", ref=cv.ref, text=cv.value)
                )

    # Pass 2, over kept rows: the column profiles.
    col_cells: dict[int, list[Cell]] = {}
    col_errors: dict[int, int] = {}
    for c in range(1, width + 1):
        cells = [grid.cell(r, c) for r in kept]
        col_errors[c] = sum(x.kind in ("error", "formula_no_value") for x in cells)
        col_cells[c] = [x for x in cells if x.kind not in ("empty", "error", "formula_no_value")]
    tied2 = {c: _tied_styles(cells) for c, cells in col_cells.items()}
    pref2 = _sheet_preference(list(tied2.values()))

    columns: list[ColumnProfile] = []
    for c in range(1, width + 1):
        cells = col_cells[c]
        dates, dm_ambiguous = _date_formats([x.raw for x in cells])
        ctype, _ = _column_type(cells, dates)
        nf = None
        if ctype == "number" and tied2[c]:
            idx, amb = _pick_style(tied2[c], pref2)
            nf = _number_format(cells, idx, amb)
        counts = Counter(x.raw for x in cells)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]
        non_null = len(cells) + col_errors[c]
        label = labels[c - 1]
        hint = None
        label_scale = _scale_of(label)
        if label_scale is not None:
            hint = ScaleHint(scale=label_scale, source="label", ref=None, text=label)
        elif ctype == "number" and title_hints:
            hint = title_hints[0]
        columns.append(
            ColumnProfile(
                col=c,
                letter=col_letter(c),
                label=label,
                type=ctype,
                non_null=non_null,
                null_rate=_rate(len(kept) - non_null, len(kept)),
                errors=col_errors[c],
                distinct=len(counts),
                unique=bool(kept) and len(cells) == len(kept) and len(counts) == len(cells),
                top=[TopValue(value=v, count=n) for v, n in top],
                samples=[CellValue(ref=x.ref, value=x.raw) for x in _spread(cells, SAMPLES_N)],
                number_format=nf,
                date_formats=dates,
                day_month_ambiguous=dm_ambiguous,
                scale_hint=hint,
            )
        )

    profile_ = Profile(
        file=Path(grid.path).name,
        sheet=grid.sheet,
        file_sha256=grid.file_sha256,
        n_rows=len(grid.rows),
        n_cols=width,
        header_rows=header_rows,
        header_found=found,
        title_rows=title_rows,
        data=DataRegion(start=start, end=end, rows=len(data_rows)),
        notes_rows=[_text_row(grid.rows[r - 1]) for r in notes],
        skip_candidates=skips,
        columns=columns,
        scale_hints=title_hints,
        unpivot_candidates=_unpivot_candidates(labels),
        keys=_keys(grid, kept, columns, col_cells),
        header_fingerprint=header_fingerprint(labels),
        warnings=warnings,
    )
    digest = hashlib.sha256(
        profile_.model_dump_json(exclude={"profile_sha256"}).encode("utf-8")
    ).hexdigest()
    return profile_.model_copy(update={"profile_sha256": digest})


def _keys(
    grid: Grid, kept: list[int], columns: list[ColumnProfile], col_cells: dict[int, list[Cell]]
) -> list[list[str]]:
    if len(kept) < 2:
        return []
    texty = [col for col in columns if col.type != "empty" and _texty(col, col_cells[col.col])]

    def name(col: ColumnProfile) -> str:
        return col.label or col.letter

    def values(col: ColumnProfile) -> list[str] | None:
        vals = []
        for r in kept:
            cell = grid.cell(r, col.col)
            v = cell.raw.strip()
            if cell.kind == "empty" or not v:
                return None
            vals.append(v)
        return vals

    vals = {col.col: values(col) for col in texty}
    singles = [col for col in texty if (v := vals[col.col]) is not None and len(set(v)) == len(v)]
    keys = [[name(col)] for col in singles]
    single_cols = {col.col for col in singles}
    pool = texty[:KEY_COLUMNS]
    for a, b in combinations(pool, 2):
        if a.col in single_cols or b.col in single_cols:
            continue
        va, vb = vals[a.col], vals[b.col]
        if va is None or vb is None:
            continue
        if len(set(zip(va, vb, strict=True))) == len(kept):
            keys.append([name(a), name(b)])
    return keys
