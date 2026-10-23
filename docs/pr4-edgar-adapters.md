# PR 4 — Source adapters, SEC EDGAR, precision-aware conflicts

Spec and work plan. Branch `feat/edgar-adapters` off `main` (54ea773, PR 3 merged). The PR
targets `main`. Sections 2–4 are FROZEN so the workstreams can build in parallel.

## 1. What this adds, and what the real data taught us

An **adapter** turns an external source into documents and facts with provenance; the store
does the rest (versioning, chunking, embedding, conflict groups, RLS). Two adapters ship:
`edgar` (SEC filings) and `local_files` (a directory a tenant supplies).

The EDGAR corpus is five life-sciences filers, **pinned to their FY2019–FY2021 10-Ks**, in two
tenants:

| Tenant | Companies (ticker · CIK) |
| --- | --- |
| `pharma-a` | Pfizer PFE 0000078003 · Gilead GILD 0000882095 · Vertex VRTX 0000875320 |
| `pharma-b` | Moderna MRNA 0001682852 · Regeneron REGN 0000872589 |

Probing the live XBRL `companyfacts` data before writing this spec found three things the
design must handle:

1. **Real restatements, sometimes more than once.** Pfizer FY2019 revenue is reported three
   ways: $51,750M (FY2019 10-K, Feb 2020), $41,172M (FY2020 10-K, Upjohn recast), $40,905M
   (FY2021 10-K). Pfizer's net income, R&D and FY2020 revenue all moved too; Gilead FY2019
   R&D $9,106M → $4,055M; Regeneron FY2019 revenue $7,863M → $6,558M and R&D $3,037M → $2,450M.
   These must land in one conflict group each, with every filing kept as a cited source.
2. **Precision is not disagreement.** Later filings repeat prior years rounded coarser:
   Vertex FY2019 revenue is `4,162,821,000` in two 10-Ks and `4,162,800,000` in the FY2021 one;
   Moderna FY2020 net loss is `-747,064,000` vs `-747,000,000`. PR 3's exact-value rule would
   flag both as conflicts. A blanket tolerance would be wrong too (it would make $1,800 and
   $1,850 rents agree). So precision becomes an explicit, per-fact attribute the adapter sets
   and the store compares at (§2).
3. **The number is not always in the 10-K document.** Pfizer's 10-K is a 266K-char shell that
   incorporates its financial statements by reference from **Exhibit 13** (782K chars, where
   "Total revenues" actually appears). The adapter ingests the primary document **and any
   EX-13**, and a fact cites the document that contains the financial statements. Also, a
   frequent filer's older 10-Ks fall out of the `submissions` "recent" window, so filings are
   resolved by **accession number** through the archive index, never by "most recent N".

Tag choice is per filing: XBRL companies switch tags over time (Moderna reports FY2019 revenue
as `RevenueFromContractWithCustomerExcludingAssessedTax` in one filing and `Revenues` in later
ones). For each `(field, period, filing)` the adapter takes the first tag in a priority list
that the filing reports; mixing tags within one filing would compare different line items.

## 2. Store changes (FROZEN) — `rag/schema.py`, `rag/store.py`

```sql
ALTER TABLE rag_facts ADD COLUMN IF NOT EXISTS precision  NUMERIC;  -- NULL = exact
ALTER TABLE rag_facts ADD COLUMN IF NOT EXISTS source_ref TEXT NOT NULL DEFAULT '';
```

`FactRow` and `Fact` gain `precision: int | None = None` and `source_ref: str = ""`.
`source_ref` is free-form provenance below the document level; the EDGAR adapter writes
`"us-gaap:<Tag>@<accession>"`.

**Conflict rule, precision-aware** (replaces "different value" in `add_facts`): two numeric
facts for the same `(entity, field, period_start, period_end)` **agree** when, with
`p = max(p_a or 1, p_b or 1)`, `round_half_up(a / p) == round_half_up(b / p)`; otherwise they
disagree. Text facts agree iff equal. A new fact that disagrees with any existing fact joins
(or creates) that group exactly as in PR 3; one that agrees with all is inserted with no group.
With both precisions NULL this is PR 3's exact rule, so the lease demo and its tests are
unchanged. Implement the comparison in Python (`fde_kit.rag.store.values_agree(a, pa, b, pb)`,
pure, exported) and keep the SQL selecting candidate rows; do not push rounding into SQL.

`query()` / `facts()` / `Fact.to_source` unchanged except the two new columns are selectable.

## 3. Adapter API (FROZEN) — `rag/adapters/`

```python
# rag/adapters/base.py
@dataclass(frozen=True)
class RawDocument:
    source: str              # stable URI, e.g. the sec.gov archive URL of the document
    title: str
    text: str                # already extracted plain text
    metadata: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RawFact:
    document_source: str     # which RawDocument (by source) this fact is cited to
    row: FactRow             # doc_id left None; ingest fills it from document_source

@dataclass(frozen=True)
class RawEntity:
    kind: str; canonical_name: str; aliases: tuple[str, ...] = ()

@dataclass
class AdapterBatch:          # one unit of ingest, e.g. one company
    entities: list[RawEntity]
    documents: list[RawDocument]
    facts: list[RawFact]
    dictionary: list[DictEntry] = field(default_factory=list)

class Adapter(Protocol):
    name: str                # "edgar", "local_files"
    def batches(self) -> Iterator[AdapterBatch]: ...

@dataclass
class IngestReport:
    documents: int; documents_changed: int; facts: int; conflict_groups: int
    entities: int; job_id: str

def ingest(session: TenantSession, adapter: Adapter, *, max_chars: int = 900) -> IngestReport:
    """For each batch: upsert entities, ingest_versioned every document (idempotent by hash),
    one extraction job per call (kind=f"adapter:{adapter.name}"), add_facts with doc_id filled
    from document_source, set_dictionary. A fact whose document_source was not in the batch is
    an error. Re-running on unchanged input writes no documents; facts are only added for
    documents whose ingest changed (so a re-run does not duplicate facts)."""
```

```python
# rag/adapters/edgar.py
@dataclass(frozen=True)
class Filing:
    cik: str; ticker: str; company: str; accession: str; form: str
    fiscal_year: int; period_end: date; filed: date

class EdgarClient:
    """HTTP to sec.gov with the contact SEC asks for, a rate limit and an on-disk cache.
    - user_agent: required, non-empty (settings.edgar_user_agent / FDE_EDGAR_USER_AGENT);
      construction raises ValueError when empty. Never logged, never written to any file.
    - max 5 requests/second (SEC allows 10; stay well under)
    - cache_dir (settings.edgar_cache_dir, default .fde/edgar-cache): GET responses cached by
      URL; `offline=True` serves only from cache and raises on a miss
    - stdlib urllib only (honours HTTPS_PROXY); no new dependency"""
    def __init__(self, user_agent: str, *, cache_dir: Path, offline: bool = False): ...
    def json(self, url: str) -> Any
    def text(self, url: str) -> str

def html_to_text(html: str) -> str
    # stdlib html.parser; drop script/style/ix:header; block tags -> newlines; table cells
    # separated by " | "; collapse whitespace. Deterministic.

def company_facts(client, cik) -> dict
def filing_documents(client, filing: Filing) -> list[tuple[str, str]]
    # (url, doc_type) for the primary 10-K document and any EX-13, from the archive index for
    # the accession (https://www.sec.gov/Archives/edgar/data/<cik>/<acc-no-dashes>/). Never
    # from submissions "recent".
def xbrl_facts(cf: dict, filing: Filing, fields: dict[str, tuple[str, ...]]) -> list[tuple[str, FactRow]]
    # (field_name, FactRow) for full-year periods (350–372 days) ending in FY2019–FY2021 that
    # THIS filing (accn) reports; first tag in the priority tuple that the filing has for that
    # period; unit USD; precision = infer_precision(val); source_ref = f"us-gaap:{tag}@{accn}"
def infer_precision(value: int) -> int
    # largest power of ten in {1, 10, ..., 10**6} that divides value; 0 -> 1. Values are
    # whole dollars in companyfacts; a later filing's $0.1M/$1M rounding shows as 10**5/10**6.

FIELDS = {
  "revenue":    ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
  "net_income": ("NetIncomeLoss",),
  "rd_expense": ("ResearchAndDevelopmentExpense",
                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
}

class EdgarAdapter:
    name = "edgar"
    def __init__(self, client: EdgarClient, filings: list[Filing], *, max_doc_chars: int | None = None)
    def batches(self) -> Iterator[AdapterBatch]
    # one batch per company: entity (kind "company", canonical = companyfacts entityName,
    # aliases = (ticker, CIK without leading zeros)); one RawDocument per filing document
    # (source = archive URL, title = f"{company} {form} FY{fy} ({doc_type})", metadata:
    # cik/accession/form/fiscal_year/doc_type/filed); facts cited to the EX-13 when the
    # filing has one, else the primary document; dictionary entries for the three fields.
    # max_doc_chars truncates text (offline fixtures); None = full text.
```

```python
# rag/adapters/manifest.py — the pinned corpus
@dataclass(frozen=True)
class ManifestEntry:
    tenant: str; ticker: str; cik: str; accession: str; form: str; fiscal_year: int
    documents: dict[str, str]     # document url -> sha256 of the FULL extracted text
EDGAR_MANIFEST: Path              # rag/adapters/edgar_manifest.json (committed)
def load_manifest() -> list[ManifestEntry]
def filings_for(tenant: str) -> list[Filing]
def check_drift(client) -> list[str]   # re-fetch, re-extract, compare hashes; human messages
```

```python
# rag/adapters/local_files.py
class LocalFilesAdapter:
    name = "local_files"
    def __init__(self, root: Path, *, glob: str = "**/*", entity: str | None = None)
    # .md/.txt read as text; .htm/.html via html_to_text; .pdf skipped with a warning (no new
    # dependency); source = path relative to root; one batch; no facts unless a sidecar
    # facts.jsonl of {"document": rel_path, "entity":..., "field":..., "value":..., "unit":...,
    # "period_start":..., "period_end":...} lines exists at root.
```

## 4. Offline fixtures (FROZEN layout)

CI and the evals must not depend on sec.gov being up. A capture script writes real, trimmed
data that the offline client serves from cache:

```
evals/fixtures/edgar/
  README.md                       # what this is, capture date, how to regenerate, trimming rules
  cache/<sha1(url)>.json|.txt     # EdgarClient cache entries (url stored inside each entry)
```

- `companyfacts` responses are trimmed to the `FIELDS` tags, USD unit, `form == "10-K"`, and
  periods ending 2019–2021. Everything kept is byte-for-byte what SEC returned for those rows.
- Document responses are stored as **extracted text truncated to 40,000 chars**; the full-text
  sha256 is in the manifest, so truncation is visible and drift is detectable.
- Archive-index responses are stored whole (small).
- Regenerate with `FDE_EDGAR_USER_AGENT="<contact>" python -m fde_kit.rag.adapters.capture`
  (writes the manifest and the fixtures). The contact is read from the environment and is never
  written to any file.

## 5. CLI (Wave 2-D)

```
fde rag ingest --tenant T --adapter edgar [--ticker PFE ...] [--offline]
fde rag ingest --tenant T --adapter local_files --dir PATH [--entity NAME]
fde rag seed-edgar [--offline]       # create pharma-a / pharma-b, ingest the manifest
fde rag edgar-drift                  # live re-fetch; report changed hashes; exit 1 on drift
```

`seed-edgar --offline` uses `evals/fixtures/edgar/cache` and needs no network and no contact.
Online commands require `FDE_EDGAR_USER_AGENT` and say so clearly when it is missing.

## 6. Evals (Wave 2-E) — `fde eval --set edgar`

Seeded with `seed-edgar --offline`; mock scripts drive the real tools under RLS, as in PR 3.

| Category | Cases | Example and checks |
| --- | --- | --- |
| `table-fact` | 3 | Moderna FY2021 revenue ($18,471M); Gilead FY2021 net income ($6,225M); Regeneron FY2021 revenue ($16,072M) — `tool_called query_table`, `contains`, `cites <filing doc url>`, `cites_only` |
| `table-conflict` | 3 | Pfizer FY2019 revenue (51,750 / 41,172 / 40,905, cites all three filings); Gilead FY2019 R&D (9,106 / 4,055); Regeneron FY2019 revenue (7,863 / 6,558) — `conflict_reported`, `cites` each filing, `cites_only` |
| `precision` | 2 | Vertex FY2019 revenue; Moderna FY2020 net loss — the tool output has **no** `conflict=true`; answer gives one value; `regex` asserts no conflict wording |
| `doc-lookup` | 2 | facts from the first 40K chars of a filing (e.g. state of incorporation, fiscal year end) — verify against the fixture text before writing the case |
| `cross-tenant` | 2 | `pharma-b` asks for Pfizer revenue; `pharma-a` asks for Moderna's — `cites_only`, not-found answer, `regex` that the other tenant's number is absent |

CI: seed-edgar --offline, then `run_evals --set edgar --provider mock --threshold 1.0`.

## 7. Work plan

### Wave 1 (parallel)

| ID | Owns | Done when |
| --- | --- | --- |
| **1-A** store | `rag/schema.py`, `rag/store.py`, `rag/__init__.py`, `tests/test_rag_facts.py` (append), `tests/test_values_agree.py` (new, pure) | §2: columns + upgrade path; `values_agree` with 15+ pure cases (Vertex and Moderna pairs agree; Pfizer, Gilead, Regeneron pairs disagree; $1,800 vs $1,850 at precision 1 disagree; text; None precisions = exact); Postgres tests: a precision-only pair forms no group, a real restatement plus a rounded repeat forms exactly one group containing the disagreeing values, the PR 3 lease conflicts unchanged |
| **1-B** adapters | `rag/adapters/{__init__,base,edgar,local_files,manifest,capture}.py`, `rag/adapters/edgar_manifest.json`, `evals/fixtures/edgar/**`, `config.py` (`edgar_user_agent`, `edgar_cache_dir`), `tests/test_adapters.py` (new) | §3–4 against the live API: capture run produces the manifest (15 filings) and fixtures; `EdgarAdapter` over the **offline** client yields 5 batches with the expected entities, documents (Pfizer filings carry an EX-13) and facts (Pfizer FY2019 revenue present from all three filings with precision and source_ref set); `html_to_text` and `infer_precision` pure tests; `ingest()` tested against a fake session (Postgres comes in Wave 2); `local_files` tests on tmp_path. Until 1-A lands, construct `FactRow` via a local shim that accepts `precision`/`source_ref` (`# TODO(integrator)`) |

### Wave 2 (parallel, after Wave 1 is merged)

| ID | Owns | Done when |
| --- | --- | --- |
| **2-C** CLI + Postgres | `cli.py` (`# --- adapters (PR 4)` region), `tests/test_edgar_ingest.py` (new, PG-gated) | §5 commands; PG tests: `seed-edgar --offline` into fresh tenants; re-run writes nothing new; conflict groups exactly as §1 (Pfizer FY2019 revenue 3 values in one group; Vertex FY2019 revenue no group); `query_table` as pharma-b sees none of pharma-a's companies; posture clean |
| **2-D** evals | `evals/golden_edgar.jsonl`, `evals/run_evals.py` (`--set edgar`, seeding helper reuse), `tests/test_evals.py`, `.github/workflows/ci.yml`, `Makefile` | §6, 12/12 locally against PG; CI steps added |
| **2-E** README | `README.md` | `### Source adapters and EDGAR` section: the three findings in §1 with the real numbers, precision rule, pinned manifest + drift, offline fixtures and how to regenerate, CLI, contact requirement; pieces/what's-not-here updated |

### Wave 3 (integrator)
Merge, swap the 1-B shim to the real `FactRow`, full suite against local Postgres, a live
`seed-edgar` (online) smoke into a scratch DB, `edgar-drift` once, push, PR → `main`, CI.

### Conflict map
`store.py`/`schema.py`: 1-A only. `rag/adapters/**`, `config.py`, fixtures: 1-B only.
`cli.py`: 2-C only. `evals/*`, `ci.yml`, `Makefile`: 2-D only. `README.md`: 2-E only.
