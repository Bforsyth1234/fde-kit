"""Column labels from header rows (§7.6 of docs/pr5-wrangle-core.md), and their fingerprint.

A column's label is its header cells top to bottom, empty ones dropped, joined with `" / "`.
Merged upper cells are filled across first, so with `Forecast` merged over C4:F4 and months in
row 5, columns C..F are `Forecast / Nov-26` .. `Forecast / Feb-27`, and a column with an empty
row-4 cell over `Site` is just `Site`.

Details this module decides (shared by profile and apply, so both see the same labels):

- **Fill across, not down.** A merged range's value (its top-left cell) is copied into every
  column of the range, in the topmost header row the range touches. Lower rows of a vertically
  merged range stay empty, so `Site` merged over A4:A5 gives `Site`, not `Site / Site`. A range
  whose top-left cell is above the header rows still fills its topmost header row.
- **Each part is whitespace-normalised**: stripped, and runs of whitespace (a wrapped header's
  newline included) become one space. Case is kept. A part that is blank after that is dropped.
- **Width** is the grid's width, so labels has one entry per column, `""` for a column with no
  header text.

Depends only on `read.py`.
"""

from __future__ import annotations

import hashlib
import re

from fde_kit.wrangle.read import Grid

LABEL_JOIN = " / "
FINGERPRINT_SEP = "\x1f"

_WS = re.compile(r"\s+")


def _clean(raw: str) -> str:
    return _WS.sub(" ", raw).strip()


def header_labels(grid: Grid, header_rows: list[int]) -> list[str]:
    """One joined label per grid column (§7.6). `header_rows` are 1-based; rows outside the grid
    read as empty."""
    width = len(grid.rows[0]) if grid.rows else 0
    rows = sorted(set(header_rows))
    if not rows or width == 0:
        return [""] * width
    parts: dict[int, list[str]] = {
        r: [_clean(grid.cell(r, c).raw) for c in range(1, width + 1)] for r in rows
    }
    for min_row, min_col, max_row, max_col in grid.merged:
        touched = [r for r in rows if min_row <= r <= max_row]
        if not touched:
            continue
        value = _clean(grid.cell(min_row, min_col).raw)
        top = touched[0]
        for c in range(min_col, min(max_col, width) + 1):
            parts[top][c - 1] = value
        for r in touched[1:]:
            for c in range(min_col, min(max_col, width) + 1):
                parts[r][c - 1] = ""
    return [LABEL_JOIN.join(p for p in (parts[r][c] for r in rows) if p) for c in range(width)]


def normalise_label(s: str) -> str:
    """Casefold and collapse whitespace (stripped)."""
    return " ".join(s.casefold().split())


def header_fingerprint(labels: list[str]) -> str:
    """First 12 hex of sha256 over the normalised labels joined by U+001F. Every column counts,
    including ones with an empty label, so adding or moving a column changes it."""
    joined = FINGERPRINT_SEP.join(normalise_label(s) for s in labels)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
