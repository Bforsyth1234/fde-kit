# PR 5 — Wrangle core: messy tables into cited facts, no LLM

Spec and work plan. Branch `feat/wrangle-core` off `main`. First of the four PRs in
[wrangle-plan.md](wrangle-plan.md). Sections 3–8 are FROZEN so the workstreams in §12 can build
in parallel. Nothing here calls a model: a person writes the mapping spec by hand, and the code
loads it. PR 7's agent will write the same spec format, so this PR is also that agent's contract.

## 1. What this adds

1. **Read any CSV or XLSX cell by cell, keeping coordinates.** Every value keeps its sheet, row
   and column from file to fact.
2. **Profile a file** deterministically. The profile covers where the table is, the header,
   totals rows, per-column types, number and date formats, scale hints, wide (month-across)
   layouts and key candidates. It's what a person reads first, and what the PR 7 agent will read.
3. **A mapping spec** (YAML): which columns are the entity, the period and the measures, and how
   each value is parsed. It uses a closed set of named ops with no expressions, and it's bound to
   its crosswalks and dictionary by one hash.
4. **Dry run and apply.** A dry run needs no database. It reports facts, rejects with cell refs
   and reasons, and fuzzy matches to review. Apply writes through the existing adapter `ingest()`
   in one transaction, and **writes nothing** if the reject rate is over the gate or the spec
   isn't approved at its current hash.
5. **Cell-level provenance.** Every fact's `source_ref` names the cell it came from, and each
   sheet is stored as a document, so `fetch_document` shows a person the row.
6. **Re-exports replace instead of piling up.** A new version of a sheet retires the facts of the
   version it supersedes, so `query_table` never sums last month's export into this month's.

**Not in this PR:** the benchmark generator (PR 6), any LLM call (PR 7), the spec library and
corrections log (PR 8). Also out: summing line-level rows into one fact (§13), PDFs, streaming
very large files, and dimensions on `rag_facts`.

## 2. Decisions taken (from wrangle-plan §9)

| Decision | Taken |
| --- | --- |
| Plant × part data | Composite entities (`DET-07/P-1234`, kind `plant_part`); no schema change |
| Spec format | YAML, validated by pydantic |
| Scale | In-memory grids; streaming waits for a real file that needs it |
| Review | Both: specs live in Git, and `fde wrangle approve` stamps an approval bound to the bundle hash |
| Real dataset | Not in PR 5 (synthetic gate is PR 6) |

## 3. Store and ingest changes (FROZEN) — `rag/store.py`, `rag/adapters/base.py`

All additive. EDGAR and `local_files` behave exactly as before, because every new parameter
defaults to the old behaviour.

```python
@dataclass
class IngestOutcome:
    doc_id: str
    version: int
    changed: bool
    content_hash: str
    previous_doc_id: str | None = None
    """The version this one superseded; None for a first version or an unchanged re-ingest."""

class TenantSession:
    def delete_facts_for_document(self, doc_id: str) -> int:
        """Delete this tenant's facts whose doc_id is `doc_id`. Returns the count. The document
        row itself stays, as provenance. Conflict groups the deleted facts were in are
        re-checked: one left with a single member, or whose remaining members all agree, is
        dissolved (conflict_group_id back to NULL)."""
```

`ingest_versioned` fills `previous_doc_id` from the `current` row it already reads.

```python
def ingest(session, adapter, *, max_chars: int = 900,
           on_new_version: Literal["keep", "replace"] = "keep") -> IngestReport: ...

@dataclass
class IngestReport:
    ...                       # unchanged fields
    facts_replaced: int = 0   # facts deleted because their document was superseded
```

- `"keep"` (default) is today's behaviour. EDGAR wants it: a filing is immutable, and a
  restatement arrives as a different document and should conflict.
- `"replace"`: when a document's ingest returns `changed` with a `previous_doc_id`, its facts are
  deleted **before** the new version's facts are added. The new facts don't conflict with their
  own previous version, but they still conflict with other sources.
- **Job kind:** `ingest` reads the optional adapter attributes `job_kind`, `job_model` and
  `job_prompt_sha256` (`getattr` with today's values as defaults) and passes them to
  `start_extraction_job`. Wrangle sets `job_kind = "wrangle:<spec>@<bundle_sha12>"`.

Why delete rather than mark: `query_table` lets the model write SQL over `rag_facts` directly. A
`superseded` flag would depend on every query remembering a `WHERE`. Deletion is safe because
the old document row (body and hash) stays, and `fde runs drift` already reports a recorded
citation to a deleted fact as `gone` and to a replaced document as `superseded`.

`tests/test_adapters.py::FakeSession` gains `delete_facts_for_document` and returns
`previous_doc_id` when the hash changes.

## 4. Project layout and files (FROZEN)

A wrangle project is a directory, normally one per customer and kept in Git:

```
acme/
  dictionary.yaml            # target fields (rag_dictionary)
  crosswalks/plants.csv      # one CSV per crosswalk
  specs/planner_forecast.yaml
```

**`dictionary.yaml`** is a list of `DictEntry`:

```yaml
- field: demand_forecast_units
  definition: Forecast customer demand for the period, in units.
  unit: units
```

**Crosswalk CSV**: header `alias,canonical`, UTF-8. Every canonical value is also matched as
itself. The crosswalk is the source of truth for the aliases of its entities. Apply upserts those
entities with exactly these aliases (`upsert_entity` replaces aliases).

**Bundle hash.** `bundle_sha256 = sha256` of the canonical JSON (sorted keys, no whitespace) of:
the spec with its `approval` block removed, plus `{name: sha256(bytes)}` for the dictionary
and each crosswalk the spec references. Approval, the job kind and every `source_ref` carry it,
so changing a crosswalk after approval invalidates the approval exactly as editing the spec
would.

## 5. MappingSpec (FROZEN) — `wrangle/spec.py`

Pydantic v2 models with `extra="forbid"`, so an unknown key is an error and not silently ignored.
Paths are relative to the spec file.

```yaml
spec: planner_forecast                # [a-z0-9_]+, unique in the project
version: 1                            # spec schema version
dictionary: ../dictionary.yaml
crosswalks:
  plants: {path: ../crosswalks/plants.csv, kind: plant}

source:
  sheet: Plan                         # XLSX: required (an XLSX file without it: SpecMismatch); CSV: omit
  header_rows: [4, 5]                 # 1-based; several are joined "Upper / Lower"
  data_starts: 6                      # default: the row after the last header row
  header_fingerprint: 9c1e2b7a55f0    # optional; apply fails on a mismatch unless --force
  document_key: "{spec}#{sheet}"     # default "{spec}/{file}#{sheet}"; see §7.4
  encoding: utf-8                     # CSV only; default utf-8-sig
  delimiter: ","                      # CSV only; default: sniffed
  skip_rows:
    - {column: Site, matches: "(?i)^(sub)?total"}
    - {blank: [Site]}
  start_after: {column: Site, matches: "^FORECAST$"}   # optional; data begins after this row
  stop_before: {column: Site, matches: "^ACTUALS"}     # optional; data ends before this row

entity:
  kind: plant_part                    # single-column: kind of that column's entities
  parts:
    - column: Site
      ops: [{op: text, strip: true}]
      resolve: {crosswalk: plants, policy: closed, fuzzy: false}
    - column: Part No
      ops: [{op: text, strip: true, case: upper}]
      resolve: {policy: open}
  template: "{0}/{1}"                 # default "{0}" for one part

period:                               # exactly one of: column, cell, unpivot
  unpivot:
    columns: {from: "Forecast / Jan-26", to: "Forecast / Dec-26"}   # joined labels (§7.2), inclusive, sheet order
    # or: matching: "^[A-Z][a-z]{2}-\\d{2}$"
    ops: [{op: regex, pattern: "Forecast / (.+)"}, {op: period, format: "%b-%y", grain: month}]
  # column: {name: Ship Date, ops: [{op: date, format: "%d/%m/%Y"}], grain: month}
  # column: {name: [YEAR, MONTH], ops: [{op: period, format: "%Y-%m", grain: month}]}  # "2024-1"
  # cell:   {ref: B2, ops: [{op: regex, pattern: "Period: (.+)"}, {op: period, format: "%b %Y", grain: month}]}

measures:
  - field: demand_forecast_units      # must be in the dictionary
    column: null                      # with unpivot: the unpivoted cells; else a column name
    ops: [{op: number, thousands: ",", decimal: ".", parens_negative: true}]
    scale: 1000                       # multiply after parsing; default 1
    precision: 1000                   # default: scale when scale > 1, else None (exact)
    unit: units                       # default: the dictionary's unit
    empty: skip                       # skip | reject; default skip
    nulls: ["--", "(S)"]              # placeholders treated as empty; default none (§7.4)

notes: |                              # free text; assumptions a reviewer should check
  Header says "(000s)". Site names match the plants crosswalk exactly.

approval:                             # written by `fde wrangle approve`, never by hand
  by: brooks
  at: 2026-10-30T15:02:11Z
  bundle_sha256: 3f2a9c1b04de…
```

Validation beyond types: unpivot excludes `measure.column` (and vice versa); every dictionary
field and crosswalk named exists; a regex compiles and has exactly one group when a later op
needs its output; `template` placeholders match `parts`; there is at most one unpivot;
`start_after`/`stop_before` regexes compile; `nulls` entries are non-blank, unpadded and
distinct; a `period.column.name` list is non-empty with no column twice.

**Bundle hash and later settings.** `start_after`, `stop_before` and `nulls` enter the bundle
hash only when set (a non-empty `nulls`), so a spec that doesn't use them hashes exactly as it
did before they existed and its approval stays valid. Setting one is an edit like any other.

**Text values.** A measure whose last op is `text` or `map` produces a text fact (`value_text`).
`precision` with a text measure is a validation error.

## 6. Ops (FROZEN) — `wrangle/parse.py`

Each value is a pipeline: the first op gets the cell's raw string, and each op passes its output
on. Failure raises `ParseError(reason)`, which becomes a reject. There are no other ops, and
no expressions.

| Op | Params | In → out | Semantics |
| --- | --- | --- | --- |
| `text` | `strip` (true), `case` (none/upper/lower), `collapse` (true) | str → str | whitespace and case only |
| `regex` | `pattern` (one group) | str → str | `re.fullmatch` unless the pattern starts with `^` or ends with an unescaped `$` (then `re.search`); `([^,]+)` and `\$(\d+)` are fullmatched; no match is a reject |
| `map` | `values: {from: to}`, `default` (absent = reject) | str → str | exact match after `text` defaults |
| `number` | `thousands` (`,` `.` `'` ` ` or none), `decimal` (`.`/`,`), `parens_negative` (false), `trailing_minus` (false), `currency` (strip `$€£¥` and ISO codes, true), `percent` (false: `12%` → reject; true → 0.12) | str → Decimal | strict: after the declared stripping, the rest must be a valid number; `1,234.5` with `thousands: "."` is a reject |
| `date` | `format` (strptime, or a list of them) **or** `excel_serial: true` | str → date | XLSX date cells arrive as ISO strings (§7.1); as a first op they skip `format` (native cells, §7.4) |
| `period` | `format` (strptime, or a list of them) **or** `pattern` like `Q{q} FY{yy}`; `grain` day/month/quarter/year/instant; `fy_start_month` (1) | str → (start, end) | month → first and last day; FY26 with `fy_start_month: 10` = 2025-10-01..2026-09-30; `instant` → start = end = the date |
| `constant` | `value` | any → str | for a fixed period or label |

Numbers stay `Decimal` through `scale`, then become `int` if integral, else stay `Decimal`.
`FactRow.value` accepts a Decimal and `add_facts` binds it as is, so it is stored exactly (a
float would go through Postgres's float8 → numeric cast, which keeps 15 significant digits and
stores `1100.0000000000002` as 1100). Rounding is never applied. JSON outputs write a Decimal
as a number when a float reads back as the same value, else as a string.

## 7. Read, profile, apply (FROZEN behaviour)

### 7.1 Grid — `wrangle/read.py`

```python
@dataclass(frozen=True)
class Cell:
    row: int; col: int        # 1-based, as Excel shows them
    raw: str                  # text as stored; numbers via repr-free str(Decimal); dates ISO
    kind: Literal["empty", "text", "number", "date", "bool", "error", "formula_no_value"]
    @property
    def ref(self) -> str: ... # "G42"

@dataclass(frozen=True)
class Grid:
    path: str; sheet: str; file_sha256: str
    rows: list[list[Cell]]
    merged: list[tuple[int, int, int, int]]
```

- XLSX via `openpyxl.load_workbook(read_only=True, data_only=True)`, then one streaming pass of
  openpyxl's sheet parser over the requested sheet only: cell values are what the full model
  gives, and the same pass records formula cells with no cached value and the sheet's
  `<mergeCells>` (a 25k-row sheet of a 19 MB workbook: 1 s and 125 MB, was 76 s and 2.8 GB).
  A formula cell whose cached value is `None` is `formula_no_value`, and is a reject wherever a
  spec reads it, **never** a 0.
  `#N/A`, `#REF!` and the like are `error`.
- In a merged range only the top-left cell has a value. `read` does **not** fill the others;
  header joining (§7.2) does that for header rows only, where it's always meant.
- CSV via the `csv` module; encoding `utf-8-sig` by default; delimiter sniffed from the first
  64 KB unless set. Every CSV cell is `text` or `empty`. Bytes that don't decode are a
  `ValueError` naming the file and byte offset and pointing at `source.encoding`.

### 7.2 Profile — `wrangle/profile.py`

`profile(grid) -> Profile`, a pydantic model serialised to JSON with `profile_sha256`. Same
input, same bytes out.

- **Header:** pick the first row `r` within the first 30 where ≥ 60% of non-empty cells are
  text, and rows `r+1..r+5` are each ≥ 60% non-text in the columns that are non-empty, or some
  column is text in row `r` and numeric in every one of them (a long table with text
  dimensions). Total-label rows and one-cell section labels are left out of that look-ahead,
  and a one-cell row over a wider table is a title, never the header. If row `r+1` is
  year-like numbers under a range merged across them in row `r`, the result is `[r, r+1]`. If
  row `r-1` has merged ranges spanning those columns (but not all of them: that is a title),
  or holds ≥ 2 text cells only in columns where row `r` is empty (a split header), it's an
  upper header and the result is
  `header_rows = [r-1, r]`. Labels are joined `"Upper / Lower"` with merged values filled across.
  Rows above are `title_rows` (kept verbatim; periods and units often live there). The full
  rules are in the module docstring.
- **Data region:** from the row after the header to the last row before 3 consecutive empty rows.
  Text below it is `notes_rows`.
- **Skip candidates:** rows whose first non-empty cell matches `(?i)^(sub|grand )?total`, and rows
  where ≥ 2 numeric cells each equal the sum of the same column over the rows since the
  previous candidate (±0.5 units). A column counts only with a non-zero cell and ≥ 2 non-zero
  cells in the window; sums are running totals, so this is linear in the rows.
- **Encoding:** `Profile.encoding` is the CSV encoding used (None for XLSX). `profile_file` /
  `fde wrangle profile` without `--encoding` try utf-8-sig, cp1252, latin-1 and warn when they
  fell back. The profile may guess; apply never does (it uses `source.encoding`).
- **Per column:** `label`, inferred type, `null_rate`, `distinct`, `unique` (bool), `top` (10
  values with counts), `samples` (10 cell refs and values), and:
  - `number_format`: thousands, decimal, parens/trailing-minus seen, currency seen, % seen
  - `date_formats`: candidate strptime formats that parse ≥ 95%, best first; `day_month_ambiguous`
    when both `%d/%m` and `%m/%d` parse every value
  - `scale_hint`: from the label or a title row: `(000s)`, `in thousands` → 1000; `$m`,
    `in millions` → 1e6
- **Wide layout:** a run of ≥ 3 adjacent header labels that all parse as periods with one format
  → `unpivot_candidates: [{from, to, format, grain}]`.
- **Keys:** single columns with `unique`, and pairs of columns unique together (checked over the
  first 10 text-ish columns only).
- **`header_fingerprint`:** the first 12 hex characters of sha256 over the normalised labels
  (casefold, whitespace collapsed) joined by `\x1f`.

The profile proposes; the spec decides. Apply never reads the profile.

### 7.3 Entity resolution — `wrangle/resolve.py`

`norm_key(s)`: casefold, NFKC, strip leading zeros from each digit run, then drop the characters
`- _ . # / ,` and spaces (zeros first, so `10-07` and `100-7` stay distinct). So `DET-07` → `det7`, `Plant 07` → `plant7`, `1007` → `1007`.

For each part with `resolve.crosswalk`:

1. exact on the cleaned value against canonicals and aliases
2. exact on `norm_key`. If one key maps to two canonicals, that is an **`ambiguous`** reject,
   and `fde wrangle check` (§9) reports the crosswalk itself as faulty.
3. if `fuzzy: true` (default false): rapidfuzz `token_set_ratio` on `norm_key`, accepted only
   when best ≥ `threshold` (default 92) **and** best − second ≥ 5. The match is used and
   written to `crosswalk.proposed.csv` for review.
4. otherwise `closed` → reject `unknown entity`; `open` → the cleaned value is its own
   canonical, and it's listed in the report as new. The store resolves names
   case-insensitively, so within one load an open value whose casefold equals an open value
   seen earlier, or a crosswalk canonical, takes that first spelling and is counted as
   `normalised` (`Widget-a`, then `WIDGET-A` → both `Widget-a`).

Parts with no crosswalk are `open`. Composite canonical = `template` filled with the resolved
canonical of each part; composites are compared casefolded the same way. False merges cost more than misses, which is why fuzzy is off by
default and the margin rule exists.

### 7.4 Apply — `wrangle/apply.py`

`run(spec_path, files) -> ApplyResult` is pure (no database). It is the dry run, and the adapter
calls it too.

Before any file is read, each file's `document_key` is rendered (§7.4, documents). Two files
with the same rendered key in one load → `SpecMismatch`: one would supersede the other within
the load.

For each file → `Grid` (the sheet named by the spec):

1. Check `header_fingerprint` if set. Mismatch → `SpecMismatch` (whole file refused, not rejects).
2. Build column labels from `header_rows`. A spec column name not found → `SpecMismatch`.
3. For each data row from `data_starts`, within the section (below): apply `skip_rows` (counted,
   not rejected). Then the entity parts → resolve; then the period. If either fails, **one
   row-level reject**, and none of that row's measures load.
4. For each measure cell (each unpivoted cell, or the measure column): an empty cell, or one
   holding a `nulls` placeholder, with `empty: skip` is counted; otherwise run `ops`, then
   `scale` → `RawFact`. Failure → **cell-level reject**.
5. **Within one document** (file + sheet), the same `(entity, field, period)` twice: equal
   values are `duplicates_dropped` (first kept). Different values both load, and `add_facts`
   puts them in a conflict group. The report names both cells. Across documents nothing is
   dropped: `add_facts` decides agreement and conflict at ingest, as between loads.

**Sections (two tables on one sheet).** `source.start_after` and `source.stop_before` are each
`{column, matches}`: the column is named by its joined label, and a row matches when that
cell's raw text matches the regex (`re.search`), the same test as a `skip_rows` match rule.
Scanning starts at `data_starts`. Rows up to and including the first `start_after` match are not
data; the data then ends before the first `stop_before` match after it. Rows outside the section
are not read (not in `rows_read`, not skipped, not rejected). A bound that matches no row is a
`SpecMismatch`: loading on without it would read the wrong table, or both. The file's report
entry gets `"section": {"start_after_row": n, "stop_before_row": m}` (only the bounds the spec
sets; absent when it sets neither). One spec per table: `stop_before: ^ADJUSTED` loads the
first, `start_after: ^ADJUSTED` the second.

**Null placeholders.** A measure's `nulls` lists strings that mean "no value" in this source
(EIA `--`; Census `(S)`, `(NA)`, `(D)`, `(X)`). A measure cell whose stripped `Cell.raw` equals
one is treated as an empty cell: with `empty: skip` it is counted per placeholder under the
file's `null_skipped` (`{"(S)": 84}`), and with `empty: reject` it is a cell-level reject with
`op: "nulls"`, reason `"null placeholder"` and the cell's raw text. Error and
formula-without-value cells are read failures first, never nulls. There is **no default list**
and nulls are per measure, not per entity or period: what's a null in one source is data in
another (`NA` is Namibia, `(D)` can be a real label, `--` a real code), and a placeholder the
spec didn't name should stay a visible reject, not vanish.

**Period from several columns.** `period.column.name` may be a list of labels. For each row,
their cells' values (whitespace-cleaned: stripped, runs collapsed) are joined with `-` (no
padding) into one text value, and the period ops run on that: `YEAR` 2024 and `MONTH` 1 give
`2024-1`, which `{op: period, format: "%Y-%m", grain: month}` reads (strptime's `%m` accepts
`1`..`12` without a leading zero). The joined value is text, so the native-cell rule doesn't
apply to it. If every part is empty the value is empty; if one is, the join (`2024-`) fails
to parse; an error or formula-without-value cell among them is a read failure. A period reject
shows the joined value as `raw`. Every source cell is in the trail: `FactRecord.period_cells`
(`("A6", "B6")`) and `period_from` `"A6+B6"`; the preview JSON carries both.

**Format lists.** `date` and `period` accept `format` as one strptime string or a list tried in
order; the first that parses wins (strptime must consume the whole value, so `%b. %Y` never
half-reads `May 2024`). Use it when one source spells labels more than one way: MRTS writes
`Jan. 2024` but `May 2024`, so `format: ["%b. %Y", "%b %Y"]`.

**Native cells.** When a pipeline's (entity part, period, measure) **first** op is `number` and
the cell's kind is `number`, the value is `Decimal(raw)` and only the remaining ops run, so a
`thousands: "."` spec reads a native 1234.5 as 1234.5. A first `date` op on a `date` cell takes
the cell's date (`date.fromisoformat` of the date part); a first `period` op on a `date` cell
expands it with `period_of(date, grain, fy_start_month)` from the op's params. Text cells, and
so every CSV cell, go through the ops as before. `apply.cell_pipeline` is the one function, and
the provenance read-back uses it too.

**Provenance (FROZEN):**

```
source_ref = "<Sheet>!<A1>@wrangle:<spec>/<bundle_sha12>/<file_sha12>"     # CSV: sheet is the file stem
```

The cell comes first, so the existing `reported_line_item` (the part before `@`) shows the cell
when `query_table` lists facts. The entity and period cells for that fact are in the report, not
the ref.

**Document per sheet:** `RawDocument(source=document_key rendered, title="<file name> — <sheet>",
text=…)`. The text's first line is `# wrangle spec=<spec> bundle=<bundle_sha256>`; the rest is
the sheet as CSV with a leading `row` column holding the Excel row number, title rows included.
That way `fetch_document` shows a person row 42 as row 42, and a corrected (re-approved) spec
over the same bytes is a new document version, so `replace` (§3) retires the wrong facts.
`metadata` holds `file_sha256`, `spec`, `bundle_sha256` and `header_rows`. `document_key`
controls versioning. Its placeholders are `{spec}` (the spec name), `{file}` (the file's base
name), `{path}` (the path as passed) and `{sheet}` (CSV: the file stem). With the default
`{spec}/{file}#{sheet}`, `Forecast_Nov.xlsx` is a different document from `Forecast_Oct.xlsx`,
two specs over one file never share a document, and the key doesn't depend on how the path was
typed. With a fixed key like `{spec}#{sheet}` (`planner_forecast#Plan`), each new export
supersedes the last, and (with `replace`, §3) retires its facts. Choose per source: snapshot
data wants a fixed key; history you want to keep as separate vintages wants distinct keys and
distinct fields.

**Report (FROZEN JSON):**

```json
{"spec": "planner_forecast", "bundle_sha256": "…", "approved": true,
 "files": [{"path": "…", "sheet": "Plan", "file_sha256": "…", "rows_read": 412,
            "rows_skipped": {"skip_rows[0]": 20, "skip_rows[1]": 3},
            "empty_skipped": 37, "null_skipped": {"--": 4}}],
 "facts": 4632, "duplicates_dropped": 0,
 "rejects": {"total": 14, "rate": 0.003,
             "by_reason": {"unknown entity": 12, "number: unparseable": 2}},
 "entities": {"exact": 388, "normalised": 22, "fuzzy": 0, "new": 0},
 "conflicts_within_load": [{"document": "planner_forecast#Plan",
                            "key": ["DET-07/P-1234", "demand_forecast_units", "2026-03"],
                            "cells": ["Plan!G42", "Plan!G97"]}]}
```

`reject rate = rejects.total / (facts + rejects.total)`. Rows skipped by rule and empty skips
are not in the denominator.

**Rejects (FROZEN JSONL)**, one per line:
`{"file", "sheet", "cell" | "row", "level": "cell"|"row", "raw", "stage": "entity"|"period"|"measure", "op", "reason"}`

### 7.5 Adapter — `wrangle/adapter.py`

```python
class WrangleAdapter:
    name = "wrangle"
    job_kind: str             # "wrangle:<spec>@<bundle_sha12>" ("wrangle-draft:…" when unapproved)
    def __init__(self, spec_path: Path, files: list[Path]): ...
    def batches(self) -> Iterator[AdapterBatch]: ...   # one batch per file
```

Each batch has: the crosswalk entities (kind from the crosswalk, aliases from the CSV), the
composite and `open` entities it produced, the sheet's document, the facts, and the dictionary
entries the spec uses. `apply` calls `ingest(session, adapter, on_new_version="replace")` inside
one `store.session(tenant)`, so the load is one transaction.

### 7.6 Resolved during Wave 1 (FROZEN)

- **Labels.** A column's label is its header cells top to bottom, empty ones dropped, joined with
  `" / "`, with merged upper cells filled across. With `header_rows: [4, 5]` and an empty row-4 cell
  over `Site`, the label is `Site`. Spec `column`, `unpivot.columns.from/to` and `blank` all
  name joined labels, and an unpivoted cell's period ops receive its joined label.
- **`period.column`.** Its ops end in `date`; `grain` (default `day`) then expands the date to the
  period containing it (`parse.period_of`). Ending in `period` is also allowed, and then `grain` must
  be absent.
- **Whitespace.** `number`, `date` and `period` ignore surrounding whitespace; no `text` op needed.
- **Defaults.** `number` defaults to `thousands: ","`, `decimal: "."`. A measure with no ops is a
  text measure; numeric measures name `number`. `crosswalks` may be omitted.
- **Reject fields.** `file` is the path as passed on the command line; `raw` is `Cell.raw` (a CSV
  cell after unquoting; `#DIV/0!` for an error cell; `""` for `formula_no_value`); `op` is the
  failing op's name, `"resolve"` for an entity that didn't resolve and `"read"` for an error or
  `formula_no_value` cell. Reasons for read failures are `"error cell"` and `"formula without
  cached value"`; others are `ParseError.reason` or `unknown entity` / `ambiguous entity`.
- **Duplicates.** "First kept" means the first row's cell is the fact's `source_ref`.
- **Later additions (real-data gaps).** `nulls`, `start_after`/`stop_before`, a `format` list and
  a `period.column.name` list (§7.4) came from loading real government files
  (`tests/fixtures/wrangle/real/`). `op` in a reject may also be `"nulls"`. The report's file
  entries always carry `null_skipped` (`{}` when nothing matched) and carry `section` only when
  the spec sets a bound. The join separator for multi-column periods is `-`
  (`spec.PERIOD_JOIN`).

## 8. Gates (FROZEN)

Apply refuses, writing nothing, and exits 1 when:

- the spec has no `approval`, or `approval.bundle_sha256` ≠ the current bundle hash, unless
  `--draft` (draft loads are tagged `wrangle-draft:` in the job kind)
- reject rate > `--max-reject-rate` (default 0.02)
- any file raises `SpecMismatch` (unless `--force` for the fingerprint check only), including two
  files whose `document_key` renders the same and an XLSX file under a spec with no
  `source.sheet`

The rejects file and the report are written either way, so a refused load always says why.

## 9. CLI — `fde wrangle`

New top-level group in `cli.py`. `profile`, `check`, `dry-run` and `approve` need only the
`wrangle` extra (which includes psycopg, because facts are built as `FactRow`s) and no database;
`apply` also needs `rag` and Postgres.

```
fde wrangle profile FILE [--sheet S] [--json]
fde wrangle check SPEC                  # validate the spec + crosswalks, print the bundle hash
fde wrangle dry-run SPEC FILE... [--out DIR]   # report.json, rejects.jsonl, facts.preview.jsonl (first 50), crosswalk.proposed.csv
fde wrangle approve SPEC --by NAME      # stamps approval; refuses if `check` fails
fde wrangle apply SPEC FILE... --tenant T [--max-reject-rate 0.02] [--draft] [--force] [--out DIR] [--url URL] [--embedder E]
```

`profile` prints a rich summary: header rows, columns with type and formats, skip candidates,
unpivot candidates, scale hints, keys. `--json` prints the model. Exit codes follow the rest of
the CLI: 0 ok, 1 refused or failed, 2 bad arguments.

## 10. Fixtures (FROZEN layout) — `tests/fixtures/wrangle/`

```
make_fixtures.py          # writes the generated files deterministically (openpyxl, csv)
project/dictionary.yaml
project/crosswalks/plants.csv            # 5 plants, aliases in 3 styles
project/specs/erp_shipments.yaml         # long CSV, one fact per row
project/specs/planner_forecast.yaml      # wide XLSX, composite entities, unpivot, (000s)
project/specs/legacy_report.yaml         # DD/MM dates, parens negatives, European decimals
erp_shipments.csv                        # generated
planner_forecast.xlsx                    # generated: 3 title rows, 2-row merged header, subtotals
legacy_report.csv                        # generated: one value disagreeing with erp_shipments
formulas.xlsx                            # COMMITTED, saved by LibreOffice so formulas have cached values;
                                         # one formula cell deliberately without one
expected/<spec>.facts.jsonl              # every expected fact with its cell
```

`formulas.xlsx` is the only committed binary. openpyxl can't write cached formula values, and
that's the case worth testing. `make_fixtures.py` documents the `soffice --headless
--convert-to xlsx` command that produced it, and the test asserts its sha256.

`real/` holds US government files (see its README; `SHA256SUMS` pins them) and
`real/project/` a dictionary and one spec per load: `state_pop`, `eia_residential`,
`eia_transport`, `county_pop`, `mrts_2024_nsa`, `mrts_2024_sa`, `generation_ct_ri`.
`tests/test_wrangle_real.py` runs each and checks **every** fact against an independent read of
the file (openpyxl or `csv`, not the wrangle reader or ops), the report's counts (0 rejects,
0 conflicts, `null_skipped`, and the 12 justified `duplicates_dropped` in MRTS NSA where NAICS
4522 and 45221 are both "Department stores"), and hand-checked anchors.

## 11. Tests

Pure (no Postgres), `tests/test_wrangle_*.py`:

- **parse:** a table of ≥ 40 cases per op; Hypothesis-free round trips for `number` (format a
  Decimal in each style, parse it back), parentheses, trailing minus, `1.234,56`, currency,
  percent; strictness (`1,234.5` with `thousands: "."` rejects); `period` month/quarter/FY with
  `fy_start_month: 10`; `instant`
- **read:** merged ranges, `formula_no_value`, error cells, CSV BOM and sniffed `;`
- **profile:** header found under title rows; two-row merged header joined; subtotal rows found
  by label and by sum; `day_month_ambiguous` true for `03/04/2026`-only columns and false once
  `13/04/2026` appears; unpivot candidate on the planner sheet; fingerprint stable across runs
- **spec:** unknown key rejected; unpivot + measure column rejected; bad template; text measure
  with precision rejected; bundle hash changes when a crosswalk byte changes and not when only
  `approval` changes
- **resolve:** each strategy; `ambiguous` on a colliding crosswalk; fuzzy margin rule (92/90 →
  reject, 95/85 → accept); closed vs open
- **apply:**
  - every fixture spec yields exactly `expected/<spec>.facts.jsonl`
  - **provenance read-back (the test that matters most):** for every fact, parse its
    `source_ref`, open the file, read that cell, run the spec's ops and scale, and assert the
    stored value
  - row-level vs cell-level rejects; within-load duplicate dropped vs conflicting pair reported
  - `SpecMismatch` on a wrong fingerprint and on a missing column
- **adapter and ingest** against `FakeSession`: `on_new_version="keep"` unchanged for existing
  adapters; `"replace"` deletes before adding; job kind taken from the adapter

Postgres (`FDE_TEST_PG_URL`), `tests/test_wrangle_pg.py`:

- apply all three fixture specs into one tenant: facts count, `query_table` returns them with
  `Plan!G42`-style line items, and `fetch_document` on a fact's doc shows the right row number
- the ERP/legacy disagreement is one conflict group; a value that agrees at the declared precision
  (`1,234` in thousands vs `1,234,400`) forms none
- **re-export:** change one cell, apply with a fixed `document_key`: one new document version,
  `facts_replaced` = the old count, the total fact count is unchanged, and the changed value is
  the only difference
- idempotent: apply twice unchanged → nothing written, and no extraction job either
- gates: unapproved, stale approval, and rate over the limit each leave the tenant's row counts
  unchanged
- isolation: tenant B sees none of tenant A's wrangle facts or documents (reuses the existing
  isolation helpers)

CI: install `.[dev]` (which gains the wrangle deps), plus one step: `fde wrangle dry-run` on each
fixture spec, then `fde wrangle apply` into a fresh tenant with the hash embedder.

## 12. Work plan

### Wave 1 (parallel)

| Stream | Files | Delivers |
| --- | --- | --- |
| **1-A** store/ingest | `rag/store.py`, `rag/adapters/base.py`, `tests/test_adapters.py` (FakeSession + new cases), `tests/test_rag_facts.py` (append) | §3 exactly; existing tests untouched and green |
| **1-B** read + parse | `wrangle/read.py`, `wrangle/parse.py`, `tests/test_wrangle_parse.py`, `tests/test_wrangle_read.py` | §6, §7.1 |
| **1-C** spec + project | `wrangle/spec.py`, `tests/test_wrangle_spec.py`, `pyproject.toml` (extra) | §4, §5, bundle hash |
| **1-D** fixtures | `tests/fixtures/wrangle/**` | §10, with expected facts written by hand from the generator's world, not from the code under test |

### Wave 2 (parallel, after Wave 1)

| Stream | Files | Delivers |
| --- | --- | --- |
| **2-A** profile | `wrangle/profile.py`, `tests/test_wrangle_profile.py` | §7.2 |
| **2-B** resolve + apply + adapter | `wrangle/resolve.py`, `wrangle/apply.py`, `wrangle/adapter.py`, `tests/test_wrangle_apply.py`, `tests/test_wrangle_resolve.py` | §7.3–7.5, §8 |

### Wave 3 (integrator)

`cli.py` (`fde wrangle …`), `wrangle/__init__.py` exports, `tests/test_wrangle_pg.py`,
`tests/test_cli_wrangle.py`, `.github/workflows/ci.yml` step, README section ("Wrangle messy
tables"), and FINDINGS: what the fixtures turned up, plus the limits in §13.

### Conflict map

Only 1-A touches existing modules. 1-C and Wave 3 both touch `pyproject.toml`: 1-C adds the
extra and Wave 3 only the CI line, so there's no overlap. `cli.py` is touched only in Wave 3.

## 13. Limits, recorded in FINDINGS when it ships

- **One cell, one fact.** Line-level extracts that need summing (40 invoice rows → one
  plant-month) aren't supported. A `sum` would need provenance to a set of cells, and that
  design belongs with the PR 6 benchmark, which will show how often it's needed.
- **Composite entities, not dimensions.** "All parts at Plant 7" in `query_table` is
  `canonical_name LIKE 'DET-07/%'`. That works, but it's a string convention. Revisit if the
  golden questions in PR 7 get awkward.
- **`add_facts` is two statements per fact.** Fine at fixture scale and at a few thousand facts;
  measure in PR 6 before batching it.
- **Each sheet is embedded as a document.** A 10k-row sheet is ~700 chunks to embed. That's
  acceptable for the hash embedder in CI; note the real cost in FINDINGS.
- **Crosswalks own their entities' aliases.** Loading a crosswalk replaces aliases another
  adapter set on the same entity. Wrangle and EDGAR entities don't overlap today.
