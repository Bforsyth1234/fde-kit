# Wrangle — messy tabular sources into facts with cell-level provenance

Plan for PRs 5–8. Status: §9 decided (as proposed); PR 5 specced in
[pr5-wrangle-core.md](pr5-wrangle-core.md); nothing built yet. Each PR is shippable on its own and leaves
`main` green; sections marked FROZEN are the contracts the later PRs build against.

## 1. Why

Step one of almost every deployment is the same: the customer's numbers live in three or four
exports that were never meant to meet. Plant `1007` in the ERP is `DET-07` in the planning
workbook and `Detroit Stamping #7` in the legacy CSV. One file reports in thousands, one in units.
One puts months down the rows, one across the columns. Dates are `20260304`, `03/04/2026` and
`4/3/26`. There are title rows above the header, a "Total" row every 20 lines, and a column whose
name means one thing in one plant and something else in another.

The kit can already store facts with provenance, keep conflicts instead of averaging them, and
cite every number an answer uses (PR 3, PR 4). What it can't do is get messy tables *into* that
layer without someone hand-writing an adapter per file. That's the gap.

The design rule for everything below:

> **The model proposes a mapping. A person approves it. Plain code applies it.**

The LLM never transforms rows. It reads a profile and samples, and writes a mapping spec: a
small, declarative, reviewable file. Deterministic code executes the spec, so a load is
reproducible, diffable and auditable, and every fact points back to the exact cell it came from.
That's what a finance or operations customer needs before they'll trust a number. It's also the
only version of "agentic data mapping" that survives an audit.

## 2. What this adds, by PR

| PR | Name | LLM? | Ships |
| --- | --- | --- | --- |
| 5 | Read, profile, spec, apply | no | `fde_kit.wrangle` core; a hand-written spec loads a messy file into facts with cell refs; rejects file; `fde wrangle profile / dry-run / apply` |
| 6 | Benchmark first | no | Seeded generator of a messy multi-source world with ground truth; scorers; a heuristic baseline mapper; `fde wrangle bench` |
| 7 | Agent proposer | yes | `fde wrangle propose`: agent + tools writes a spec, self-checks with dry runs; LLM adjudication for the entity-match residue; Q&A golden set; replay fixtures for CI |
| 8 | Mapping library and reuse | yes | Approved specs reused by header fingerprint; human corrections logged and fed back; second-customer experiment; FINDINGS section |

The benchmark lands **before** the agent on purpose. The agent's value is measured against a
non-LLM baseline on data with known answers, not judged by eye.

Not in scope: PDFs and scanned tables (the `local_files` PDF gap stays), streaming or very large
files (see §9), writing back to source systems, and forecasting itself. Wrangle produces the
clean, cited facts a forecast or an agent reads. It doesn't forecast.

## 3. Module layout

```
src/fde_kit/wrangle/
  __init__.py
  read.py        CSV / XLSX -> Grid (cells with sheet, row, col; raw string + cell type)
  profile.py     Grid -> Profile (deterministic; header detection, per-column stats, patterns)
  parse.py       the closed set of value parsers and transforms (numbers, dates, text)
  spec.py        MappingSpec (pydantic), load / dump / sha256, validation against a Profile
  apply.py       (spec, Grid) -> ApplyResult: entities, RawFacts, rejects, report
  resolve.py     entity matching: exact -> normalised -> fuzzy -> (PR 7) LLM adjudication
  joins.py       key-overlap discovery between columns of different sources
  adapter.py     WrangleAdapter: spec + files -> AdapterBatch (plugs into the existing ingest)
  propose.py     (PR 7) proposer agent and its tools
  library.py     (PR 8) spec library, header fingerprints, corrections log
src/fde_kit/prompts/wrangle_propose.md        (PR 7)
evals/wrangle/generator.py, scorers.py, bench.py, golden_wrangle.jsonl   (PR 6, 7)
tests/test_wrangle_*.py
```

New optional extra, so the core install stays slim:

```toml
wrangle = ["openpyxl>=3.1", "pyyaml>=6.0", "rapidfuzz>=3.6"]
```

No pandas. Grids are read cell by cell so every value keeps its coordinates, and the files this
targets (exports, workbooks) fit in memory comfortably. If that stops being true, §9 says what
changes.

## 4. PR 5 — read, profile, spec, apply

### 4.1 Grid (FROZEN)

```python
@dataclass(frozen=True)
class Cell:
    sheet: str        # "" for CSV
    row: int          # 1-based, as a person sees it in Excel
    col: int          # 1-based
    raw: str          # exactly what the file holds, as text
    kind: str         # "empty" | "text" | "number" | "date" | "bool" | "formula_value"

@dataclass(frozen=True)
class Grid:
    path: str
    sheet: str
    file_sha256: str
    cells: list[list[Cell]]
    merged: list[tuple[int, int, int, int]]   # XLSX merged ranges, (r1, c1, r2, c2)
```

`Cell.ref` renders the A1 form (`Sheet1!G42`). Formulas are read as their cached values
(`data_only=True`); a formula cell with no cached value is a reject, never a silent zero.

### 4.2 Profile

Deterministic and hashed (`profile_sha256`). It's what the proposer reads and what a person
reads to sanity-check a spec.

- **Table region:** header row(s) detected as the first row(s) where most cells are text and the
  rows below are type-consistent. Title rows above the header and notes below the data are
  recorded and excluded. Two-row headers (a merged group label over sub-labels) are joined as
  `Group / Sub`.
- **Skip candidates:** rows whose first non-empty cell matches `total|subtotal|grand total` or
  whose numeric cells equal the sum of the rows above them.
- **Per column:** inferred type, null rate, distinct count, uniqueness, top values, 10 samples,
  and patterns:
  - dates: candidate formats ranked; **day/month ambiguity** flagged unless a value > 12 settles
    it (`03/04/2026` alone is reported as ambiguous, never guessed)
  - numbers: thousands separator, decimal mark (`1.234,56`), parentheses negatives, trailing
    minus, currency symbol, `%`
  - unit and scale hints from the header (`($000)`, `in thousands`, `MT`, `kg`, `USD m`)
  - **wide layout:** a run of headers that all parse as periods (`Jan-26 … Dec-26`, `Q1 FY26`)
    is flagged as an unpivot candidate
- **Key-likeness:** uniqueness alone and in pairs, so composite keys (`plant + part`) surface.

### 4.3 MappingSpec (FROZEN)

YAML on disk, validated by pydantic, hashed on its canonical JSON form. Example:

```yaml
spec: planner_forecast
version: 1
source:
  match: {glob: "*Forecast*.xlsx", sheet: "Plan", header_fingerprint: "9c1e…"}
  header_rows: [4, 5]                # two-row header, joined "Group / Sub"
  skip_rows:
    - {column: "A", matches: "^(Sub)?[Tt]otal"}
entity:
  kind: plant
  from: ["Site"]                     # one or more columns; several -> composite key
  template: "{Site}"
  resolve: {crosswalk: plants, policy: closed}   # closed: unknown names are rejects
period:
  unpivot:
    columns: {from: "Jan-26", to: "Dec-26"}
    parse: {op: period, format: "%b-%y", grain: month}
measures:
  - field: demand_forecast_units     # must exist in rag_dictionary
    value: {op: number, decimal: ".", thousands: ",", parens_negative: true}
    scale: 1000                      # header said "(000s)"
    precision: 1000                  # feeds values_agree, so 1,234 (000s) agrees with 1,234,400
    unit: units
provenance:
  proposed_by: {model: "", prompt_sha256: "", profile_sha256: "…"}   # filled by PR 7
  approved_by: ""                    # a name; apply refuses an unapproved spec unless --draft
```

The transforms are a **closed set** of named ops in `parse.py`: `number`, `date`, `period`,
`text` (strip, case, collapse whitespace), `map` (lookup table), `regex` (extract one group),
`concat`, `coalesce`, `constant`. No expressions and no eval. A spec the model writes can't run
code, and every op has property tests.

### 4.4 Apply

`apply(spec, grid) -> ApplyResult`:

- **facts:** `RawFact`s whose `row.source_ref` is `wrangle:<spec>@<spec_sha12>/<file_sha12>/<Sheet>!<A1>`.
  For unpivoted measures it's the value cell; the entity and period cells are listed in the
  rejects/report trail, not crammed into the ref.
- **document:** each sheet is also ingested as a `RawDocument` (body = the sheet rendered as CSV
  with Excel row numbers), so `fetch_document` can show a person the row a number came from.
  Versioning by content hash, idempotent re-runs and `fde runs drift` all work unchanged.
- **extraction job:** `kind = "wrangle:<spec>@<spec_sha12>"`, with `model` and
  `prompt_sha256` when the spec was proposed. No schema change.
- **rejects:** `rejects.jsonl`, one line per cell or row that didn't load, with the cell ref, the
  raw value, the op that failed and why (`ambiguous date`, `unknown entity 'Detroit #9'`,
  `formula without cached value`, `non-numeric in measure`).
- **report:** rows read, rows skipped by rule, facts produced, rejects by reason, entities
  matched by strategy, conflicts created.
- **Gate:** `--max-reject-rate` (default 2%). Above it, nothing is written. A load that is 30%
  rejects is a wrong spec, not a dirty file.

`WrangleAdapter(specs, files)` wraps this in the existing `Adapter` protocol, so `ingest()`
does entities, documents, facts and the dictionary exactly as it does for EDGAR. Cross-source
disagreement (plant 7, March, two different shipment numbers) lands in a conflict group through
the existing precision-aware rule, with no new code.

### 4.5 Entity resolution (non-LLM part)

`resolve.py` matches in order and records the strategy and score on every match:

1. exact against canonical names and aliases
2. normalised: case, whitespace, punctuation, leading zeros, common prefixes (`Plant 07` ↔ `07`)
3. fuzzy (rapidfuzz token ratio) above a high threshold, **only within a closed crosswalk**

A crosswalk is a reviewed CSV (`alias,canonical,kind`) loaded as entity aliases. With
`policy: closed`, anything unmatched is a reject. With `policy: open`, a new entity is created
and listed in the report. **False merges cost more than misses** (two plants' numbers silently
summed is the worst outcome here), so thresholds are tuned for precision and the benchmark
weights them that way.

### 4.6 CLI

```
fde wrangle profile FILE... [--json]
fde wrangle dry-run SPEC FILE...            # report + first 20 facts + rejects; writes nothing
fde wrangle apply SPEC FILE... --tenant T [--max-reject-rate 0.02] [--draft]
```

### 4.7 Tests

- parser property tests: `number`/`date` round-trip over generated formats; parentheses,
  trailing minus, European decimals; ambiguous dates are rejected, never guessed
- **provenance read-back:** for every fact, open the file, read the cell at `source_ref`, parse
  it with the spec's op, and assert that it equals the stored value. This is the one test that
  matters most.
- header detection on title rows, two-row merged headers, notes under the table
- unpivot of a wide sheet; totals rows skipped and not double-counted
- reject gate: above the threshold nothing is written (checked in the DB)
- idempotency: apply twice, no new facts; change one cell, a new document version whose facts
  replace the previous version's (see [pr5-wrangle-core.md](pr5-wrangle-core.md) §3), so totals
  never double-count
- two sources, one disagreeing value: one conflict group with both facts; in-precision
  agreement creates no group
- RLS: wrangle facts for tenant A are invisible to B (reuses the isolation harness)

## 5. PR 6 — benchmark first

### 5.1 Generator

`evals/wrangle/generator.py --seed N` builds a ground-truth world and renders it into messy
sources. Every injected mess is logged with an id, so scores can be broken down by mess type.

**World:** 20 plants, ~300 parts, 24 months. Per plant-month: units shipped, revenue, COGS,
on-hand inventory. Per plant-part-month: demand forecast. One trial balance per plant with local
GL account codes.

**Sources (each a different "system"):**

| Source | Shape | Messes injected |
| --- | --- | --- |
| A — ERP extract (CSV) | long; one row per plant-part-month | numeric plant codes (`1007`), `YYYYMMDD` dates, units, duplicate rows, trailing spaces |
| B — planner workbook (XLSX) | wide; months across columns | title rows, two-row merged header, `(000s)` scale, subtotal rows every plant, names like `Detroit Stamping #7` |
| C — legacy report (CSV) | long; plant-month | `DET-07` codes, `DD/MM/YYYY` with some ambiguous days, parentheses negatives for returns, European decimals in one region, typos in plant names, **values that disagree with A** |
| D — local trial balances (XLSX, one sheet per plant) | GL lines | local account codes and names to be mapped onto a standard chart of accounts (a *value* crosswalk, not a column mapping) |

Ground truth is written next to the sources: the correct spec for each, the plant crosswalk, the
CoA crosswalk, the full fact table, and the list of injected conflicts.

### 5.2 Scorers

| Metric | What it checks |
| --- | --- |
| Column mapping P / R | source columns → dictionary fields, scale and unit right |
| Entity resolution P / R | **false merges reported separately** and weighted ×5 |
| CoA mapping accuracy | local accounts → standard accounts |
| Value accuracy | loaded facts vs truth, compared with `values_agree` at the declared precision |
| Provenance accuracy | each fact's cell ref, read back, yields that fact (§4.7, run over the whole world) |
| Conflict recall / precision | injected disagreements flagged; no conflicts invented |
| Reject correctness | rejects are cells that really can't be parsed or resolved |
| Human edit distance | fields changed between a proposed spec and the truth spec (PR 7 onward) |
| Cost | tokens, wall time, LLM calls per source |

### 5.3 Baseline

`HeuristicMapper`: header synonym dictionary + fuzzy header match + the PR 5 profile heuristics.
No LLM. It sets the floor the agent has to beat. If it scores close to the agent on some mess
type, that goes in FINDINGS too.

`fde wrangle bench [--seed N] [--mapper heuristic|agent] [--provider mock|live]` prints the
table and writes JSON, like `fde eval`. CI runs the heuristic baseline at a fixed seed and gates
on its scores not regressing.

## 6. PR 7 — the agent proposer

### 6.1 Loop

`fde wrangle propose FILE --tenant T --target TARGET.yaml --out SPEC.yaml`

The target names the dictionary fields and entity kinds wanted. The proposer is an ordinary kit
`Agent` run (traced, recorded, replayable) with prompt `wrangle_propose.md` and these tools:

| Tool | Returns |
| --- | --- |
| `profile_source(file, sheet)` | the §4.2 profile |
| `sample_rows(file, sheet, start, n)` | up to 30 rows with cell refs |
| `column_values(file, sheet, column, n)` | distinct values with counts |
| `dictionary()` / `entities(kind)` | the target fields; known canonical names and aliases |
| `check_join(a, b, normalise)` | value overlap between two key columns: containment both ways, unmatched examples |
| `dry_run(spec_yaml)` | validation errors, or the apply report: facts, rejects by reason, 10 sample facts with cell refs |

The agent iterates until `dry_run` is clean and the reject rate is under the gate, or it hits the
turn budget. It must state every assumption it couldn't verify (ambiguous dates, guessed scale)
in a `notes:` block on the spec, and that block is shown at review. Output is a spec with
`approved_by: ""`. **Nothing reaches the database until a person approves it.**

It sees the profile, samples and distinct values, never whole files. That keeps the cost flat in
file size and limits how much customer data reaches the model. Tenant runs already redact
content in traces.

### 6.2 LLM adjudication for entities

After §4.5, the leftover names (typically 2–5%) go to one batched call. Each gets a shortlist of
the top 5 candidates by fuzzy score, and the model answers with a candidate or `none`, plus a
reason. Answers are written to the crosswalk as **proposed** rows for review, not applied.

### 6.3 Evals

- `fde wrangle bench --mapper agent --provider live` over 3 seeds, compared with the baseline
  per mess type
- `evals/golden_wrangle.jsonl`, about 12 cases run through the existing harness after loading
  the benchmark world: "What did Plant 7 ship in March 2026?" (`cites` the cell), "Do the ERP and
  the legacy report agree on Plant 12's Q2 returns?" (must report the conflict and cite both),
  "What was total COGS on the standard chart for the Ohio plants?" (`query_table`), and one
  cross-tenant refusal
- proposer runs recorded and committed as replay fixtures, so CI runs the whole thing with no
  API key, like the existing replay cases

## 7. PR 8 — mapping library and reuse

The point of an FDE kit is that the tenth deployment is cheaper than the first.

- **Fingerprint reuse:** an approved spec stores a fingerprint of the normalised header row(s).
  A new file with the same fingerprint gets the spec directly: zero LLM calls, dry run, review.
  Monthly re-exports are the common case.
- **Nearest-spec retrieval:** with no exact match, the 3 most similar approved specs (header
  similarity) go to the proposer as examples.
- **Corrections log:** `fde wrangle approve SPEC` diffs the proposed spec against the approved
  one and appends each changed field to `corrections.jsonl` (what the agent said, what the person
  changed it to, which mess type). Recurring corrections become rules in the proposer prompt,
  and the prompt hash changes, so the effect is measurable.
- **Isolation:** library entries are per tenant by default. A *shared* pattern library is allowed
  to hold header names, units and date formats **only, never cell values**. A lint check enforces
  this, with a test that seeds a tenant value and asserts it can't reach the shared library.
- **Experiment for FINDINGS:** generate customer 1 and customer 2 (similar systems, different
  plants, different messes). Measure human edit distance, LLM calls and time to an approved spec
  for customer 2 with and without the library. That's the number to quote.

## 8. Order of work and size

| PR | Rough size | Depends on | Done when |
| --- | --- | --- | --- |
| 5 | ~1,500 lines + tests | `main` | provenance read-back passes on all fixture files; reject gate holds; CI green |
| 6 | ~800 | 5 | baseline scores published in FINDINGS; CI gates on them at a fixed seed |
| 7 | ~900 + prompt + fixtures | 5, 6 | agent beats the baseline on column mapping and entity P/R over 3 seeds; golden_wrangle passes live and in replay |
| 8 | ~600 | 7 | customer-2 experiment written up with numbers |

PR 5 is the long pole and carries most of the value on its own: a person can hand-write a spec
today and get cited, conflict-aware facts out of a messy workbook.

## 9. Decisions to make before PR 5

1. **Composite entities vs dimensions.** `rag_facts` has one entity per fact. Plant × part ×
   month needs either composite entities (`DET-07/P-1234`, kind `plant_part`, no schema change)
   or a `dims JSONB` column on facts. *Proposal: composite for now*, and record it as a limit in
   FINDINGS. Revisit if `query_table` questions across parts get awkward.
2. **Spec format.** YAML (readable in review, needs pyyaml) vs JSON (stdlib). *Proposal: YAML.*
3. **Scale.** In-memory grids are fine to a few million cells. Past that, `read.py` streams rows
   (`openpyxl` read-only mode, the csv module) and `apply` works in chunks. Not built until a real
   file needs it.
4. **Where the review happens.** The spec file in Git (a PR is the review) vs a `fde wrangle
   approve` command that stamps `approved_by`. *Proposal: both.* The command writes the stamp
   and the corrections log; Git holds the history.
5. **A real public dataset alongside the synthetic one?** Synthetic gives exact ground truth;
   a public messy dataset would add credibility to FINDINGS. *Proposal: synthetic is the gate;
   one public set as a write-up only, in PR 8.*
