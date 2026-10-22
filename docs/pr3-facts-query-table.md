# PR 3 — Provenance, facts, `query_table` and `fetch_document`

Spec and work plan. Branch `feat/facts-query-table` off `feat/record-replay-fork` (stacked on
PR 2; rebases onto `main` cleanly once #6 and #8 merge). Sections 2–4 are FROZEN so the
workstreams can build in parallel. Everything new lives under `fde_kit.rag` and inherits the
forced-RLS / partition-per-tenant / posture-check model unchanged.

## 1. What this adds

Three things the kit cannot do today:

1. **Prove what a citation pointed at.** `rag_documents` gets `content_hash`, `version`,
   `fetched_at`, `body`; `ingest()` becomes idempotent by hash and versions a changed source
   instead of silently replacing it. `fetch_document` reads a slice of the stored body, so the
   agent can start from a short passage and go deeper without blowing context.
2. **Reach the numbers, not just the prose.** A harmonised fact layer (`rag_entities`,
   `rag_facts`, `rag_dictionary`, `rag_extraction_jobs`) and a `query_table` tool that runs
   read-only SQL under the tenant's RLS session. Every row returned carries `fact_id`,
   `doc_id`, `chunk_id`, which become `Source`s.
3. **Keep conflicts, don't average them.** Two facts for the same `(entity, field, period)`
   share a `conflict_group_id` and both come back, flagged. The model is told to report the
   disagreement; the eval checks that it did.

Not in this PR: EDGAR / adapters (PR 4), LLM extraction jobs (phase 2 — the table exists, the
command does not), replaying inside Temporal.

## 2. Schema (FROZEN)

All four new tables are `PARTITION BY LIST (tenant_id)`, added to `TENANT_TABLES` and
`PARTITIONED_TABLES` in `schema.py`, so `migrate` applies the forced policy and grants,
`create_tenant` makes their partitions, and `check_posture` audits them with no further code.

```sql
-- rag_documents: add (ALTER ... ADD COLUMN IF NOT EXISTS for existing DBs)
content_hash  TEXT NOT NULL DEFAULT ''      -- sha256 of body
version       INT  NOT NULL DEFAULT 1
fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now()
superseded_by UUID                          -- doc_id of the next version, NULL = current
body          TEXT NOT NULL DEFAULT ''      -- full text, what fetch_document slices
-- UNIQUE (tenant_id, source) becomes UNIQUE (tenant_id, source, version)
-- (migrate: DROP CONSTRAINT IF EXISTS rag_documents_tenant_id_source_key; ADD CONSTRAINT
--  rag_documents_source_version_key UNIQUE (tenant_id, source, version))

-- rag_chunks: add
char_start INT, char_end INT                -- offsets of the chunk body inside documents.body,
                                            -- NULL when not locatable (title-prefixed chunks
                                            -- are located by their body after the title line)

CREATE TABLE IF NOT EXISTS rag_entities (
    entity_id       UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                  -- 'company', 'property', 'drug', ...
    canonical_name  TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{}',
    PRIMARY KEY (tenant_id, entity_id),
    UNIQUE (tenant_id, kind, canonical_name)
) PARTITION BY LIST (tenant_id);

CREATE TABLE IF NOT EXISTS rag_extraction_jobs (
    extraction_job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                  -- 'seed', 'adapter:edgar', 'llm'
    model           TEXT NOT NULL DEFAULT '',
    prompt_sha256   TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, extraction_job_id)
) PARTITION BY LIST (tenant_id);

CREATE TABLE IF NOT EXISTS rag_facts (
    fact_id         UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    entity_id       UUID NOT NULL,
    field           TEXT NOT NULL,                  -- key into rag_dictionary
    period_start    DATE,
    period_end      DATE,
    value_num       NUMERIC,                        -- exactly one of value_num / value_text
    value_text      TEXT,
    unit            TEXT NOT NULL DEFAULT '',
    doc_id          UUID,                           -- provenance: the document it came from
    chunk_id        UUID,                           -- and the chunk, when known
    conflict_group_id UUID,                         -- NULL = no disagreement on record
    extraction_job_id UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, fact_id),
    FOREIGN KEY (tenant_id, entity_id) REFERENCES rag_entities (tenant_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, doc_id)    REFERENCES rag_documents (tenant_id, doc_id) ON DELETE SET NULL,
    FOREIGN KEY (tenant_id, chunk_id)  REFERENCES rag_chunks (tenant_id, chunk_id) ON DELETE SET NULL,
    FOREIGN KEY (tenant_id, extraction_job_id) REFERENCES rag_extraction_jobs (tenant_id, extraction_job_id) ON DELETE SET NULL,
    CHECK ((value_num IS NULL) <> (value_text IS NULL))
) PARTITION BY LIST (tenant_id);
CREATE INDEX IF NOT EXISTS rag_facts_lookup_idx ON rag_facts (tenant_id, entity_id, field, period_start, period_end);

CREATE TABLE IF NOT EXISTS rag_dictionary (
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    field           TEXT NOT NULL,
    definition      TEXT NOT NULL,
    unit            TEXT NOT NULL DEFAULT '',
    allowed_values  TEXT[] NOT NULL DEFAULT '{}',
    notes           TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant_id, field)
) PARTITION BY LIST (tenant_id);
```

Composite foreign keys everywhere: a fact can only point at its own tenant's entity, document
and chunk, exactly as `rag_chunks → rag_documents` does today. The constraint, not the policy,
enforces it.

Versioning semantics in `ingest()`:

- same `source`, same `sha256(text)` → no-op; returns the current `doc_id` (`changed=False`).
- same `source`, new hash → insert `version = max+1`, set the previous current row's
  `superseded_by` to the new `doc_id`, **delete the previous version's chunks** (and this
  tenant's cache). The old document row stays as provenance (source, hash, version, body),
  so an old citation's `doc_id` + `content_hash` still resolves; its chunk ids do not. Search
  only ever sees current chunks. `fetch_document` on a superseded doc returns its body with
  `superseded_by` set, so a reader can see what changed.
- `reset` in `demo_corpus.seed` keeps working: `DELETE FROM rag_documents` cascades.

## 3. Store API (FROZEN) — `store.py`

```python
@dataclass
class IngestOutcome:
    doc_id: str
    version: int
    changed: bool            # False = identical hash, nothing written
    content_hash: str

@dataclass
class DocSlice:
    doc_id: str; source: str; title: str; version: int; content_hash: str
    superseded_by: str | None
    char_start: int; char_end: int; text: str
    total_chars: int
    next: int | None         # char offset to continue from, None at end

@dataclass
class DictEntry:
    field: str; definition: str; unit: str = ""; allowed_values: tuple[str, ...] = (); notes: str = ""

@dataclass
class FactRow:               # INPUT shape for add_facts
    entity: str              # canonical name or alias; resolved to entity_id (error if unknown)
    field: str
    value: float | int | str # number -> value_num, str -> value_text
    unit: str = ""
    period_start: date | None = None
    period_end: date | None = None
    doc_id: str | None = None
    chunk_id: str | None = None

@dataclass
class Fact:                  # OUTPUT shape
    fact_id: str; entity_id: str; entity: str; field: str
    period_start: date | None; period_end: date | None
    value: float | str; unit: str
    doc_id: str | None; chunk_id: str | None
    doc_source: str | None; doc_hash: str | None
    conflict_group_id: str | None
    extraction_job_id: str | None
    def to_source(self, tenant_id: str | None = None) -> Source
        # source_id=fact_id, uri=doc_source or f"fact:{fact_id}", title=f"{entity} {field}",
        # doc_id=doc_id, content_hash=doc_hash, span=f"fact:{fact_id}", tenant_id=tenant_id

@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool          # more than max_rows existed
    elapsed_ms: float

class SqlRejected(ValueError):
    """query_table refused the statement before sending it: not a single SELECT, or it
    touches something off-limits. `.reason` is a short human string."""

class TenantSession:
    # existing, changed semantics (returns doc_id as before; outcome on .last_ingest):
    def ingest(self, source, text, *, title="", metadata=None, max_chars=900) -> str
    last_ingest: IngestOutcome | None
    def ingest_versioned(self, source, text, *, title="", metadata=None, max_chars=900) -> IngestOutcome

    # documents
    def fetch(self, doc_id: str, char_start: int = 0, char_end: int = 6000) -> DocSlice
        # clamps the range; raises KeyError for an unknown doc_id (RLS makes another tenant's
        # doc_id unknown, so the error is identical for "no such doc" and "not yours")

    # entities / facts / dictionary
    def upsert_entity(self, kind: str, canonical_name: str, aliases: Iterable[str] = ()) -> str
    def resolve_entity(self, name: str) -> str | None       # canonical or alias, case-insensitive
    def start_extraction_job(self, kind: str, *, model: str = "", prompt_sha256: str = "") -> str
    def add_facts(self, rows: list[FactRow], *, extraction_job_id: str | None = None) -> list[str]
        # returns fact_ids. CONFLICT RULE: for each row, existing facts with the same
        # (entity_id, field, period_start, period_end) and a different value (num or text)
        # → all of them and the new row get the same conflict_group_id (the existing group's
        # id if one exists, else a new uuid). Same value → a second row is still inserted
        # (two sources agreeing is evidence, not a duplicate) but no conflict group.
    def facts(self, entity: str | None = None, field: str | None = None) -> list[Fact]
    def set_dictionary(self, entries: Iterable[DictEntry]) -> None   # upsert by field
    def dictionary(self) -> list[DictEntry]

    # read-only SQL
    def query(self, sql: str, *, max_rows: int = 200, timeout_ms: int = 5000) -> QueryResult
```

`query()` contract:

1. Guard (pure, in `rag/sqlguard.py`, unit-testable without Postgres):
   `check_select(sql) -> str` returns the cleaned statement or raises `SqlRejected`. Rules:
   strip comments and trailing `;`; must match `^\s*(with\b.*?)?select\b` (one statement,
   no second `;`); reject, case-insensitive, any of the tokens `insert|update|delete|merge|
   truncate|drop|alter|create|grant|revoke|copy|vacuum|analyze|set|reset|show|listen|notify|
   do|call|execute|prepare|lock|security|pg_|information_schema|current_setting|set_config|
   into\s+(temp|temporary|unlogged)?\s*\w` as whole words, and `\$\$`. The guard is
   honest about what it is: a keyword filter. The real boundary is the next two points.
2. Execution, inside the tenant session's transaction: `SET LOCAL transaction_read_only = on`
   (switching to read-only mid-transaction is allowed; switching back is not), `SET LOCAL
   statement_timeout = <timeout_ms>`, then `SELECT * FROM (<sql>) AS q LIMIT <max_rows+1>`.
   The model writes SQL against the parent tables `rag_facts`, `rag_entities`,
   `rag_documents`, `rag_dictionary` (stable names; RLS applies through the parents). Row
   `max_rows+1` present → `truncated=True`, trimmed.
3. Any `psycopg` error → re-raised as `SqlRejected(reason=<pg message first line>)`, so the
   tool returns it to the model as `is_error` and the run continues.

## 4. Tools (FROZEN) — `tool.py`

`tenant_search_registry(store, tenant_id, registry=None, *, cached=True, tools=("search_documents", "fetch_document", "query_table"))`
— same function, now registers up to three tools. `tools=("search_documents",)` reproduces
today's behaviour exactly (the isolation suite's existing tests keep passing unchanged).

```python
def search_documents(query: str, k: int = 5) -> ToolResult      # as in PR 1

def fetch_document(doc_id: str, char_start: int = 0, char_end: int = 6000) -> ToolResult
    """Read part of one document by its doc_id (from a search result). Returns the text
    and, if more remains, `next: <offset>`; call again with char_start=<offset>."""
    # content: "[#1] {title} ({source}) v{version} chars {start}-{end} of {total}\n{text}\n\n
    #           next: {next}" (last line only when next is not None)
    # sources: (Source(source_id=f"{doc_id}:{start}-{end}", uri=source, title=title,
    #           doc_id=doc_id, content_hash=content_hash, span=f"chars:{start}-{end}", tenant_id),)
    # unknown doc_id -> ToolResult("No document with that id.", is_error=True)

def query_table(sql: str) -> ToolResult
    """Run one read-only SQL SELECT over this customer's fact tables. <DICTIONARY> is appended
    to this description at registry build time (see below). Rows marked conflict=true
    disagree with another source: report both values and say they conflict."""
    # content: a compact table, one row per line, columns from QueryResult plus a leading
    #   "[#i]" marker per row, plus a header line; rows in a conflict group get " conflict=true";
    #   "... (truncated at 200 rows; add a WHERE or LIMIT)" when truncated
    # sources: one Source per returned row THAT HAS a fact_id column (to_source on the fact);
    #   rows without fact_id (e.g. SELECT count(*)) get no Source and no [#i]
    # SqlRejected -> ToolResult(f"Query rejected: {reason}", is_error=True)
```

Dictionary in the description: after registering, the builder sets
`out.get("query_table").description = base_doc + "\n\nTables: rag_facts(entity_id, field, period_start, period_end, value_num, value_text, unit, doc_id, chunk_id, conflict_group_id), rag_entities(entity_id, kind, canonical_name, aliases), rag_documents(doc_id, source, title, version). Join facts to entities on entity_id. Fields:\n" + one line per DictEntry "- {field} ({unit}): {definition}"`. Built once per registry from `session.dictionary()`, so it costs one query per request, not per call.

A `conflict_reported` eval check (in `evals/scorers.py`, 2-E) passes when the output contains
every value in `check.value` (a list of strings) and one of the words "conflict", "conflicts",
"disagree", "differs", "restated".

## 5. Demo data — `demo_corpus.py`

Facts from the lease corpus, so the attack suite and the evals have real, tenant-shaped
numbers:

- Entities: one `property_manager` entity per tenant (canonical = `TENANTS[t]["name"]`,
  aliases = `[tenant_id, short name]`), one `unit` entity per lease.
- Facts per tenant (field, value, unit, from the lease docs with `doc_id`): `late_fee`
  (USD or `percent_of_rent` for oakline), `grace_days` (days), `notice_days` (days),
  `pet_deposit` (USD, or value_text "not applicable"), `security_deposit_months` (months,
  1.5 / 1 / 2); per unit: `monthly_rent` (USD), `lease_start`/`lease_end` as `period_*` on a
  `monthly_rent` row.
- **One deliberate conflict per tenant:** an addendum document (`<tenant>/addenda/late-fee-2026.md`)
  that revises the late fee (harbor $75 → $85, summit $50 → $60, oakline 5% → 6%); both the
  lease value and the addendum value are loaded as facts for the same `(entity, late_fee,
  period 2026-01-01..NULL)` so they land in one conflict group. Searching "late fee" must
  still return the tenant's own numbers only (the isolation suite's shared-template property
  is preserved: every tenant has the same addendum text with a different number).
- Dictionary: one `DictEntry` per field above, units named.
- `seed()` loads entities, facts, dictionary after the documents; returns chunk counts as
  before plus a `facts` count per tenant (return type stays `dict[str, int]`; facts count
  under key `f"{tid}:facts"`).

## 6. Work plan

### Wave 1 (parallel, contract-first)

| ID | Workstream | Owns | Done when |
| --- | --- | --- | --- |
| **1-A** | Schema + store | `rag/schema.py`, `rag/store.py`, `rag/sqlguard.py` (new), `rag/__init__.py` exports, `tests/test_sqlguard.py` (new, pure), `tests/test_rag_facts.py` (new, Postgres-gated like the isolation suite) | §2 DDL + ALTERs in `DDL`/`migrate`; `TENANT_TABLES`/`PARTITIONED_TABLES` extended; §3 API complete; `check_select` guard with 20+ pure tests (accepts CTEs, rejects every token, strips comments, rejects two statements); Postgres tests for versioning (no-op, new version, old chunks gone, fetch on superseded), entities/aliases, add_facts conflict rule, dictionary upsert, `query()` read-only (an `UPDATE` smuggled past the guard still fails on `transaction_read_only`), timeout, truncation, and cross-tenant: tenant A's `query("SELECT * FROM rag_facts")` returns only A's rows and `fetch(B's doc_id)` raises KeyError. `ruff` clean. You cannot run the Postgres tests locally; write them carefully against the isolation suite's fixtures and expect CI to be the first run |
| **1-B** | Tools | `rag/tool.py`, `tests/test_rag_tool.py` (new, no Postgres: a fake session object implementing §3's `fetch`, `query`, `dictionary`, `cached_search`/`search`) | §4 exactly: three tools, `tools=` selection, description built from the dictionary, `[#i]` per row only when the row has a `fact_id` column, `conflict=true` marker, truncation note, `next:` line, error paths return `is_error` ToolResults, no tool has a tenant argument; model-supplied `tenant_id` / `filters` dropped (schema has no such field) |
| **1-C** | Demo data | `rag/demo_corpus.py`, `tests/test_demo_corpus.py` (new, pure: shapes/counts/conflict presence, no Postgres) | §5: addendum docs, `entities(tid)`, `facts(tid)` returning `FactRow`s, `dictionary()`; `seed()` extended; the addendum text shares a template across tenants with only the number differing; pure tests assert every tenant has exactly one conflicting late-fee pair and that per-unit rent facts carry periods |

Wave 1 contract notes: 1-B and 1-C import `FactRow`, `DictEntry`, `DocSlice`, `QueryResult`,
`SqlRejected` from `fde_kit.rag.store` — those names exist only once 1-A lands. Until then,
define identical dataclasses in a local `_contract.py` next to your tests / module and import
from there; the integrator replaces the import line at merge. Keep field names exactly as §3.

### Wave 2 (parallel, after Wave 1 is merged)

| ID | Workstream | Owns | Done when |
| --- | --- | --- | --- |
| **2-D** | Attack suite + CLI | `tests/test_rag_isolation.py` (append a `# --- facts, query_table, fetch_document (PR 3)` section), `cli.py` (`fde rag query --tenant T "sql"`, `fde rag fetch --tenant T DOC_ID [--start --end]`, `fde rag dictionary --tenant T`, `fde rag ask` uses the three-tool registry) | New isolation tests, all against the seeded demo: query_table as A over every parent table returns only A's rows; hostile SQL (`UNION SELECT ... FROM rag_facts WHERE tenant_id='harbor'`, `; DELETE`, `pg_class`, `set_config`, `COPY`, a CTE that updates) is rejected or returns nothing from B; `fetch_document(B's doc_id)` as A errors; `query_table` can't write even when the guard is bypassed in a test (`session.conn.execute` directly after `transaction_read_only`); conflict rows carry the marker; the posture check reports the four new tables and refuses when one of their policies is weakened (mutation test); timing: query plans touch only A's partitions. README's attack table gets these rows (2-F) |
| **2-E** | Evals | `evals/golden_rag.jsonl` (new), `evals/run_evals.py` (`--set rag`, per-case `tenant`, registry from the store), `evals/scorers.py` (`conflict_reported`), `.github/workflows/ci.yml` (step), `Makefile` (`rag-agent-eval`), `tests/test_evals.py` | 12 cases over the lease corpus: `doc-lookup` 4 (`cites` the lease source), `table-fact` 4 (`tool_called query_table`, `contains` value, `cites` the fact's document), `table-conflict` 2 (`conflict_reported` both values, `cites` both docs), `cross-tenant` 2 (A asks for B's unit; `cites_only`, output says not found). Every case has a `mock` script. `fde eval --set rag --provider mock` runs in CI after `fde rag seed-demo` with `FDE_RAG_ADMIN_URL`; pass threshold 1.0 |
| **2-F** | README | `README.md` | New `### Facts and query_table` subsection under Multi-tenant retrieval: schema in a table, versioning rules, the three tools, the guard's honesty paragraph ("a keyword filter; the boundary is transaction_read_only + RLS"), conflict rule; attack-suite table rows for 2-D's tests; pieces table / what's-not-here updated; `fde rag query/fetch/dictionary` in the CLI block |

### Wave 3 (integrator)

| ID | Step |
| --- | --- |
| **3-G** | Merge A, swap B/C `_contract` imports to `fde_kit.rag.store`, merge B, C; `ruff`; pure tests |
| **3-H** | Push, let CI run the Postgres suites (first real run of the schema); iterate on failures here; `fde eval --provider mock` 25/25 unchanged |
| **3-I** | PR |

### Conflict map

- `rag/store.py`, `rag/schema.py`: 1-A only. `rag/tool.py`: 1-B only. `rag/demo_corpus.py`: 1-C only.
- `cli.py`: 2-D only (a `# --- rag facts (2-D)` region inside `cmd_rag` and the `rag` parser).
- `evals/*`, `ci.yml`, `Makefile`: 2-E only. `README.md`: 2-F only.
- `tests/test_rag_isolation.py`: 2-D appends only.
