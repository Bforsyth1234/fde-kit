# Wrangle fixtures (PR 5, §10)

A tiny, deterministic world of 5 plants, 3 parts and a few months, written out as three messy
source files plus one LibreOffice-saved workbook. `make_fixtures.py` holds the world (the ground
truth). It writes each source file from the world with its messes injected, and it writes
`expected/*.jsonl` from the world too, never by parsing the files back. The expected files are
an independent oracle for `wrangle.apply`.

## Regenerate

```
python tests/fixtures/wrangle/make_fixtures.py              # CSVs, planner XLSX, expected/
python tests/fixtures/wrangle/make_fixtures.py --formulas   # also rebuild formulas.xlsx (soffice)
```

The plain run is idempotent: the same bytes every time, because XLSX zips get a fixed timestamp
and fixed `dcterms:created/modified`. It also checks `formulas.xlsx` against `FORMULAS_SHA256`
and exits 1 on a mismatch. `--formulas` rebuilds that file with LibreOffice and prints the new
sha256, which you then paste into `FORMULAS_SHA256`. The output differs across LibreOffice
versions, so the file is committed and never rebuilt in CI. The procedure is in the module
docstring.

## The world

| Canonical | ERP code (`erp_shipments`) | Legacy code (`legacy_report`) | Planner site (`planner_forecast`) |
| --- | --- | --- | --- |
| Detroit Stamping 07 | 1007 | DET-07 | Detroit Plt 07 |
| Toledo Assembly 12 | 1012 | TOL-12 | Toledo Plt 12 |
| Saltillo Castings 21 | 1021 | SAL-21 | Saltillo Plt 21 |
| Puebla Machining 33 | 1033 | PUE-33 | Puebla Plt 33 |
| Monterrey Paint 45 | 1045 | MTY-45 | Monterrey Plt 45 |

`project/crosswalks/plants.csv` lists all 15 aliases. Parts are P-1234, P-2210 and P-3307.

| Source | Months | Field(s) |
| --- | --- | --- |
| `erp_shipments.csv` | Oct 2025 – Sep 2026, every plant | `shipped_units` |
| `legacy_report.csv` | Apr – Sep 2025, every plant, plus Oct 2025 for Detroit and Toledo | `shipped_units`, `returned_units`, `revenue_eur` |
| `planner_forecast.xlsx` | Nov-26 – Feb-27, 9 plant/part pairs | `demand_forecast_units` (entity kind `plant_part`, e.g. `Detroit Stamping 07/P-1234`) |
| `formulas.xlsx` | Mar 2026 | `line_total_usd` (entity kind `item`) |

## Files and injected messes

**`erp_shipments.csv`** is a long CSV with one row per plant-month. Header
`PLANT_CD,SHIP_DT,QTY_SHIPPED`, comma-delimited, no BOM. Spec: `project/specs/erp_shipments.yaml`.
- Plant codes are ERP style (`1007`), and Saltillo's code has a trailing space (`"1021 "`).
- Dates are `YYYYMMDD` month-end posting dates, read at month grain.
- Puebla quantities carry a thousands comma and are quoted (`"62,600"`). Monterrey quantities
  have trailing spaces.
- **One exact duplicate row**: Saltillo Feb 2026 (row 24) is sent again as the last row (row 62).
  It should count as `duplicates_dropped: 1` with the first row kept, so the expected fact cites
  C24.
- **One unparseable number**: Toledo Mar 2026, `C28 = "38O37"` (a letter O typed for a zero).
  That's one cell-level reject, and there is no fact for that plant-month.

**`planner_forecast.xlsx`**, sheet `Plan` (plus a decoy `Notes` sheet). Spec:
`project/specs/planner_forecast.yaml`.
- 3 title rows: a title, then `Units (000s)` in A2, then an export note.
- A two-row header (rows 4–5). `Forecast` is merged across C4:F4, above `Nov-26 Dec-26 Jan-27
  Feb-27` in row 5. `Site`, `Part No` and a trailing `Total` column have no upper label.
- The layout is wide: unpivot four month columns, across a year boundary.
- Values are in thousands, as numbers with at most a quarter fraction (exact in binary).
  Puebla P-1234 Dec-26 (D14) is a number stored as text, `"1,012.5"`.
- A `Subtotal` row follows each site, and a `Total` row ends the sheet. All are skipped by
  `skip_rows`.
- Toledo's site is typed in capitals, `TOLEDO PLT 12`. It matches the crosswalk only through
  `norm_key` (step 2 in §7.3), so `entities.normalised > 0`.
- One part is lower case (`p-2210`, fixed by `case: upper`) and one is padded (`" P-3307 "`,
  fixed by `strip`). Part No is an `open` part.
- One empty cell, Monterrey P-3307 Feb-27 (F18). That's `empty_skipped: 1`, with no fact and no
  reject.
- The fixed `document_key: "planner-forecast#{sheet}"` means a re-export supersedes the last.

**`legacy_report.csv`** is a European export. Spec: `project/specs/legacy_report.yaml`.
- UTF-8 with a BOM, a `;` delimiter, `.` thousands and `,` decimal throughout
  (`1.091.331,53`).
- Plant codes are legacy style (`DET-07`).
- Report dates are `DD/MM/YYYY` month-ends, so every day is > 12 (`30/04/2025`) and the format
  isn't day/month ambiguous.
- `Shipped (000s)` is in thousands with one decimal (`46,5`), so `scale` and `precision` are
  1000.
- `Returns` shows returns as negatives in parentheses: `(40)`, and `(1.204)` for Puebla Jun 2025.
- **One unknown plant**: `GRR-03` on row 20 isn't in the crosswalk. That's one row-level reject
  (stage `entity`), and none of its three measures load.

**`formulas.xlsx`** is COMMITTED. It was saved by LibreOffice, so the formulas have cached
values. Sheet `Calc`: `A1 = "Period: Mar 2026"`, header in row 2, items in rows 3–6, and a
`Total` row 7 with `=SUM(D3:D5)`.
- D3 and D5 have cached values (240 and 1000).
- **D4 (`=B4*C4`) has its cached `<v>` stripped**, so it reads as `formula_no_value`. It must
  be a reject, never a 0.
- D6 is `=B6/C6` with C6 = 0, cached as the error `#DIV/0!`, so it's an `error` cell.
- Spec `project/specs/formulas.yaml` is optional, for dry-run and read tests. Its reject rate
  is 50%, so `apply` refuses it under the default gate.

## The ERP / legacy overlap

The only facts that share an `(entity, field, period)` across sources are the Oct 2025
`shipped_units` facts for two plants. They can be compared with `values_agree`: ERP at
precision None, legacy at precision 1000, so p = 1000.

| Plant | ERP (exact) | Legacy (000s) | Result |
| --- | --- | --- | --- |
| Detroit Stamping 07 | 48613 (`erp_shipments!C2`) | `48,6` → 48600 (`legacy_report!C33`) | **agree** (49 = 49): no conflict group |
| Toledo Assembly 12 | 36942 (`erp_shipments!C3`) | `39,9` → 39900 (`legacy_report!C34`) | **disagree** (37 ≠ 40): **one conflict group** |

`make_fixtures.py::_check_overlap` asserts this. Nothing else overlaps: the legacy history ends
where the ERP begins, and the planner field is a different field.

## `expected/`

| File | Lines |
| --- | --- |
| `erp_shipments.facts.jsonl` | 59 (60 plant-months − 1 unparseable; the duplicate appears once) |
| `erp_shipments.rejects.jsonl` | 1 (cell-level, C28) |
| `planner_forecast.facts.jsonl` | 35 (9 pairs × 4 months − 1 empty) |
| `legacy_report.facts.jsonl` | 96 (32 known rows × 3 measures) |
| `legacy_report.rejects.jsonl` | 1 (row-level, row 20) |
| `formulas.facts.jsonl` | 2 |
| `formulas.rejects.jsonl` | 2 (D4 `formula_no_value`, D6 error) |

Expected reject rates under §7.4 are erp 1/60 ≈ 1.7%, legacy 1/97 ≈ 1.0% and planner 0. All
three are under the default 2% gate.

**Facts** have one JSON object per line, with the keys `entity, field, value, unit,
period_start, period_end, precision, source_ref`. Values are after `scale`, ints when integral
(§6). Periods are ISO dates. The lines come in file row order, then column order (unpivot) or
measure order, but compare them as a set.

**`source_ref` is the cell part only**, `"<Sheet>!<A1>"`, where a CSV's sheet is its file stem
(for example `erp_shipments!C2`, `Plan!D14`). The full FROZEN form is
`<Sheet>!<A1>@wrangle:<spec>/<bundle_sha12>/<file_sha12>`, and the bundle and file hashes
aren't knowable before the loader exists. Tests should compare `source_ref.split("@")[0]`, or
build the suffix from the computed bundle hash and the file's sha256.

**Rejects** use the §7.4 shape, minus fields that can't be known in advance. `cell` or `row`,
`level`, `stage` and `file`/`sheet` are firm. `file` is the base name, so a test should compare
it against `Path(file).name`. `raw` and `op` are present only where they are unambiguous, and
`reason` is the expected wording but advisory. Tests should match firmly on `(sheet,
cell|row, level, stage)`.
