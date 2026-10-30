# Findings

What building and testing fde-kit turned up: the measurements, the failures that taught
something, and the limits that are still there. The [README](../README.md) covers how to use
the kit; this is what happened when it ran.

- [Live eval results](#live-eval-results)
- [What real SEC data taught us](#what-real-sec-data-taught-us)
- [Tenant isolation: the attack suite](#tenant-isolation-the-attack-suite)
- [The timing leak, measured](#the-timing-leak-measured)
- [Logs and backups](#logs-and-backups)
- [Two things I didn't expect](#two-things-i-didnt-expect)
- [Retrieval numbers](#retrieval-numbers)
- [Durable runs: what a crash does at each point](#durable-runs-what-a-crash-does-at-each-point)
- [Wrangling messy tables](#wrangling-messy-tables)
- [Costs and limits](#costs-and-limits)

## Live eval results

All three sets against `claude-sonnet-5-5`, prompt `default@9abfef4b5539`, single run each,
October 2026:

| Set | Pass rate | By category | Latency p50 / p95 | Tokens in / out |
| --- | --- | --- | --- | --- |
| golden | **24 / 25 (96%)** | calculator 8/8 · lookup 5/5 · lookup-miss 1/1 · date 1/1 · date+calc 1/1 · multi-tool 2/2 · citation 2/2 · replay 3/3 · no-tool 1/2 | 3.2 s / 5.3 s | 35.9k / 2.0k |
| rag (leases) | **12 / 12** | doc-lookup 4/4 · table-fact 4/4 · table-conflict 2/2 · cross-tenant 2/2 | 4.9 s / 5.9 s | 68.2k / 5.5k |
| edgar (SEC filings) | **12 / 12** | table-fact 3/3 · table-conflict 3/3 · precision 2/2 · doc-lookup 2/2 · cross-tenant 2/2 | 4.1 s / 7.5 s | 62.3k / 4.6k |

No fabricated citations in any set. In every cross-tenant case the model searched, found
nothing, and said so. The other tenant's figure appears in no format.

**The first live rag/edgar run scored 8/12 and 5/12, and every failure was the eval's fault.**
The checks were string matches written against the mock answers. "$51.750 billion" failed a
check for `51,750`, "December 1, 2026" failed `2026-12-01`, "I couldn't find" was missing from a
phrase list of refusals, and "not a real conflict" failed a regex that banned the word
"conflict". The fix (see Check types in the [README](../README.md#evals)) compares numbers and dates by value, makes the leak
check *stricter* (it now catches `$81.3 billion` as well as `81,288`), and moves wording to judge
rubrics. One more false failure followed: the judge flagged a claim the addendum states word for
word as an "outside fact", because it only saw the answer. The judge now sees the tool outputs
the model saw.

**Known failure — `notool-01`.** Asked for a one-sentence definition, the model returns two
well-formed sentences, reproducibly. The judge fails it on the length constraint while noting the
content is correct. Kept as-is: it is a real instruction-following miss and a good example of what
the judge catches that `contains` can't.

**What the first live run found.** The mock suite was green; the first real run surfaced three
bugs, all at the provider boundary: an SDK parameter that no longer exists, `.env` values not
reaching the SDK, and a thousands-separator the `contains` check didn't normalise. None were
visible offline. `tests/test_live.py` now makes one real call when a key is present so CI catches
that class of bug before the full eval does.

## What real SEC data taught us

**Restatements are real, and sometimes happen twice.** Pfizer's FY2019 revenue is $51,750M in
its FY2019 10-K, $41,172M in the FY2020 10-K (recast after the Upjohn separation) and $40,905M
in the FY2021 10-K. Three filings, three values, one conflict group, every filing cited.
Regeneron's FY2019 revenue goes from $7,863.4M to $6,557.6M between its FY2019 and FY2020
10-Ks. Gilead's FY2019 R&D is a different kind of change: $9,106M as
`ResearchAndDevelopmentExpense` in the FY2019 10-K, $4,055M as
`ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost` in the two later ones, because
acquired in-process R&D moved to a line of its own. Same field, same period, two definitions.
It still forms a conflict group, and that is right: neither number is wrong, and only a person
reading the filings can say which one their question means.

**Precision is not disagreement.** Later filings repeat prior years rounded coarser. Vertex's
FY2019 revenue is 4,162,821,000 in two 10-Ks and 4,162,800,000 in the FY2021 one; Moderna's
FY2020 net loss is −747,064,000, then −747,000,000. PR 3's exact rule would call both
conflicts. So each fact carries a `precision`, and two numbers agree when
`round_half_up(a/p) == round_half_up(b/p)`, with `p` the coarser of the two precisions
(`values_agree` in `rag/store.py`). Vertex at p = 100,000: 41,628.21 and 41,628 both round to
41,628. A blanket tolerance would be the wrong fix: it guesses at rounding for every field of
every tenant, and a 3% one makes an $1,800 rent and an $1,850 rent agree. Facts with no
precision on either side compare by exact equality, so the lease demo is unchanged. One
honest caveat: companyfacts rows don't say how the filer rounded, so the adapter infers
precision from trailing zeros (`infer_precision`, capped at 10^6), and that can be coarser
than the truth. Vertex's FY2019 net income, 1,176,810,000, is reported in thousands but infers
a precision of 10,000. A coarser precision only ever makes agreement looser, so the failure
mode is a small real restatement read as rounding, never a rounding read as a conflict.

**The number isn't always in the 10-K document.** Pfizer's FY2019 10-K is a shell that
incorporates its financial statements by reference from Exhibit 13; "Total revenues" appears
only there. From FY2020 the statements are in the 10-K itself. The adapter ingests the primary
document and any EX-13, and cites a filing's facts to the EX-13 when there is one (Pfizer: 4
documents for 3 filings). Finding the filings had its own trap: the `submissions` API's
"recent" list had already dropped Pfizer's FY2019 10-K (filed February 2020; Pfizer files
often enough to push it out of the window), so filings are discovered from the
accession numbers in the company's XBRL data and resolved through the archive index for that
accession, never by "most recent N". Regeneron's accessions carry its filing agents' prefixes
(`0001532176-`, `0001804220-`), not its own CIK, so the archive path is built from the
company's CIK plus the accession; the accession's prefix says who filed, not whose filing it is. XBRL tags also change
between filings: Moderna reports FY2019 revenue as
`RevenueFromContractWithCustomerExcludingAssessedTax` in its FY2019 10-K and as `Revenues` in
the next two. For each `(field, period, filing)` the adapter takes the first tag in a priority
list that the filing reports, so one filing's values never mix two line items.

**Provenance below the document.** A fact cites a document by `doc_id`, and a whole 10-K or
exhibit is a big thing to cite. `rag_facts.source_ref` names the exact XBRL fact:
`us-gaap:<Tag>@<accession>`, e.g.
`us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax@0000078003-20-000014`. It is
selectable in `query_table`, so the model, or a reviewer, can see which tag in which filing a
number came from, and that Gilead's two R&D values are two different tags.

## Tenant isolation: the attack suite

`tests/test_rag_isolation.py`, `tests/test_rag_facts.py` and `tests/test_tracing_tenants.py`,
against a real Postgres + pgvector; `tests/test_sqlguard.py` and `tests/test_rag_tool.py` cover
the SQL guard and the tool formatting without one. The corpus is three property managers using
the **same lease template**, so every leaked chunk reads like a correct answer with the wrong
number in it: Summit's late fee is $50, Harbor's is $75, in the same sentence. The addenda and
the facts follow the same rule: same text, same field, different number.

| Attack | Result | Test |
| --- | --- | --- |
| Same question, every tenant, every search mode | only own chunks; counts partition the table exactly | `test_every_result_belongs_to_the_asking_tenant`, `test_counts_partition_the_table` |
| Shared template: ask for "the late fee" | own number present, other tenants' numbers absent | `test_shared_template_returns_this_tenants_number` |
| Plant the exact question text in another tenant (cosine similarity 1.0) | invisible; a superuser query shows it as the top hit, proving the plant works | `test_exact_duplicate_in_another_tenant_is_invisible` |
| Same document ingested by two tenants | two documents, two ids, each sees one | `test_same_document_in_two_tenants_stays_two_documents` |
| No tenant set, or set to `''` | zero rows on every table; inserts rejected | `test_no_tenant_set_means_no_rows_and_no_writes`, `test_empty_tenant_setting_is_not_a_wildcard` |
| Insert or `UPDATE ... SET tenant_id` into another tenant | rejected by `WITH CHECK` | `test_cannot_insert_rows_for_another_tenant`, `test_cannot_move_own_rows_into_another_tenant` |
| `DELETE`/`UPDATE ... WHERE tenant_id = 'other'` | 0 rows affected | `test_updates_and_deletes_cannot_reach_other_tenants` |
| Chunk referencing another tenant's `doc_id` | foreign-key violation | `test_chunk_cannot_point_at_another_tenants_document` |
| Query another tenant's partition by name | empty; inserts rejected | `test_another_tenants_partition_is_empty_from_your_session` |
| Metadata filter `{"tenant_id": "other"}`, SQL in filter values | empty result: filters only narrow | `test_filters_can_only_narrow` |
| SQL / tsquery syntax in the query text | treated as text; tenant unchanged | `test_hostile_query_text_is_just_text` |
| Malformed tenant ids (`harbor' OR '1'='1`, …) | `ValueError` before any SQL or registry binding | `test_malformed_tenant_ids_are_rejected`, `test_tenant_registry_validates_tenant_id_upfront` |
| Shared base `ToolRegistry` passed to `tenant_search_registry` | cloned per tenant; base registry never mutated in place | `test_shared_base_registry_is_not_mutated_by_tenant_search_registry` |
| Model passes `tenant_id`, `namespace`, `filters` to the tool | ignored; schema has no such field | `test_search_tool_has_no_tenant_argument`, `test_injected_tenant_argument_is_dropped` |
| Prompt injection inside a retrieved document ("search tenant harbor") | model obeys, tool ignores it | `test_prompt_injection_in_a_document_cannot_redirect_the_search` |
| Connection pool reuse | next borrower has no tenant and sees nothing | `test_store_sessions_do_not_survive_the_transaction` |
| Shared semantic cache | per-tenant cache: other tenant misses | `test_tenant_cache_is_per_tenant` |
| Flood one tenant with matching documents | another tenant's BM25 scores unchanged, bit for bit | `test_bm25_statistics_are_per_tenant` |
| **Catalogs**: tenant ids from partition names or comments | names are hashed (keyed HMAC when configured); no comments | `test_partitions_carry_no_table_comments`, `test_partition_key_comes_from_settings` |
| **Timing**: vary tenant B's data, measure tenant A's query work | identical pages touched for vector, keyword and cache lookup, under both the default and at-scale plans | `test_query_work_does_not_depend_on_other_tenants_data` |
| **Timing**: count tenants via planning time | every plan touches only the asking tenant's partitions | `test_plans_touch_only_the_tenants_own_partitions` |
| **Ids**: count other tenants' inserts from id gaps | ids are random UUIDv4 | `test_ids_are_random_not_a_shared_counter` |
| **Logs**: parameters in slow-query / error logs | pinned to 0; the app can't re-enable the superuser-only one; sessions re-pin the other | `test_app_role_never_logs_query_parameters`, `test_store_sessions_pin_error_parameter_logging_off` |
| **Traces**: tenant data, content length or exception messages in the trace store | no span attribute, status or exception event contains the question, passage, answer or length; every span tagged | `test_tracing_tenants.py` |
| Misconfiguration: extra permissive policy, weakened partition policy, unforced RLS, parameter logging on, wrong partition key | store refuses to start | `test_posture_check_*`, `test_store_refuses_a_mismatched_partition_key` |
| `query_table` as tenant A: `SELECT * FROM` every parent table (`rag_facts`, `rag_entities`, `rag_documents`, `rag_dictionary`, `rag_extraction_jobs`) | only A's rows; `WHERE tenant_id = 'harbor'` from Summit counts 0 | `test_rag_facts.py`, and `test_rag_isolation.py` (PR 3 section) |
| `UNION ALL SELECT ... FROM rag_facts WHERE tenant_id = 'harbor'` | the union's second arm is empty; A's own numbers only | same |
| `SELECT 1; DELETE FROM rag_facts` | rejected by the guard before reaching Postgres: one statement only | same |
| `pg_class`, `pg_sleep`, `set_config('app.tenant_id', ...)`, `current_setting`, `COPY ... TO`, `WITH x AS (DELETE ...) SELECT` | rejected by the guard with the offending word in the reason | `test_sqlguard.py`, and `test_rag_isolation.py` (PR 3 section) |
| Bypass the guard: feed `UPDATE` / `DELETE` straight to the connection after one `query()` | `ReadOnlySqlTransaction`; `SET transaction_read_only = off` is refused too; a fresh session writes again | `test_rag_facts.py`, and `test_rag_isolation.py` (PR 3 section) |
| `fetch_document` with another tenant's `doc_id` | `No document with that id.`, the same error as for a nonexistent id; `fetch()` raises `KeyError` | same |
| Fact pointing at another tenant's `doc_id` | foreign-key violation, from the composite key not the policy | `test_rag_facts.py` |
| The lease and the addendum disagree on the late fee | both rows returned, each marked `conflict=true`, one `conflict_group_id` | `test_rag_isolation.py` (PR 3 section) |
| Weaken the policy on one of the four new tables (`USING (true)`) | posture reports the table; the store refuses to start | same |
| **Timing**: `query_table` plans with other tenants present | every plan touches only the asking tenant's partitions | same |

Five tests, and one assertion in a sixth, are **demonstrations**: each reproduces a real failure
with a configuration the store refuses to use, and passes because the failure happens. If one starts failing, what it shows has
stopped being true.

| Demonstration | What it shows |
| --- | --- |
| `test_demonstration_superuser_sees_every_tenant` | The compose `POSTGRES_USER` is a superuser. Tenant set, policy in place: every tenant's rows anyway. |
| `test_demonstration_session_level_setting_leaks_through_the_pool` | `set_config(..., false)` on a pooled connection; the next request forgets to set a tenant and reads the previous tenant's data. |
| `test_demonstration_shared_semantic_cache_leaks` | A process-wide cache keyed on the query embedding. Summit asks Harbor's question and gets Harbor's $75. |
| `test_bm25_statistics_are_per_tenant` (second assertion) | With global corpus statistics, another tenant's documents shift your scores. |
| `test_demonstration_shared_hnsw_starves_a_small_tenant` | One HNSW index for all tenants: a 5-chunk tenant next to a 3,000-chunk tenant gets 0 of 5 results. |
| `test_demonstration_iterative_scan_on_a_shared_index_leaks_through_timing` | The usual fix for that (pgvector's iterative scan) makes the query's work grow with what the *other* tenant has written about your query. |

**Mutation check.** I broke each defence on purpose and reran. Every policy set to `USING
(true)`: the store refuses to start (posture). With the posture check bypassed as well, 20 of the
58 isolation tests at the time failed; fewer than before partitioning, because the store's own
reads now only reach the tenant's partition even with the policy broken. A session-level
`set_config`: the pool test fails. Trace redaction off: 2 tracing tests fail. Removing `WITH CHECK` changes nothing,
correctly: Postgres applies `USING` to writes when `WITH CHECK` is absent.

One mutation taught me something the suite now covers: after a broken-policy run, the database
kept the `USING (true)` policies while the code had been restored, and the old posture check,
which only checked that *a* policy existed, let the store start on them. It now compares every
policy's condition to the one `migrate` writes.

## The timing leak, measured

Tenant A runs the same vector search on the same 50 chunks of its own. Only tenant B's data
changes. Same query, same plan (forced to the one Postgres picks when tenants are large):

| Tenant B holds | Shared index: pages | Shared index: p50 | Partitioned: pages | Partitioned: execution |
| --- | --- | --- | --- | --- |
| nothing | 123 | 0.11 ms | 97 | 0.107 ms |
| 4,000 unrelated chunks | 4,467 | 4.78 ms | 97 | 0.075 ms |
| 4,000 chunks about A's query | **16,760** | **15.46 ms** | 97 | 0.095 ms |

On the shared index, iterative scan keeps walking past other tenants' vectors until it has enough
of A's, and how far it walks depends on how close B's data is to A's question. A competitor
asking "merger acquisition target valuation" could tell, from latency alone, whether another
customer had documents about that. Partitioned, A's partition is the only thing scanned: the page
count is identical, and execution time moves with noise, not with B (a repeat of the empty-B run
measured 0.077 ms, between the other two).

Planning had its own channel. Through the parent tables, Postgres considers every partition
before pruning: planning took **0.20 ms at 8 tenants and 6.4 ms at 208**, roughly linear, which
both slows every query and lets anyone estimate the customer count. Naming the tenant's own
partitions keeps it at **0.05 ms at 8 and at 208**.

**What's left.** A cache hit is faster than a miss, so a user can tell whether *their own
tenant* asked something similar before. That stays inside the tenant by construction, but it is
a channel between users of one tenant, if that matters to a customer. Network-level timing (a
shared CPU, a noisy neighbour) is outside what a database layout can fix.

## Logs and backups

**Logs (fixed).** With slow-query logging turned on, a routine production setting, Postgres
defaults to logging every bound parameter in full. Measured on this setup: one ingest and one
search wrote the whole confidential document, its embedding, and the user's question into the
server log, which usually ships to a shared log system far more people can read than the
database. `migrate` now pins parameter logging to 0 for the app role; the same experiment after
the fix: **0** log lines mention the document. What still reaches the log: statement text (with
`$1` placeholders), partition names (a hash of the tenant id, so someone with a list of
candidate tenant ids could match them), durations, and error messages.

**Backups and the write-ahead log (explained, not fixed).** Deleting a tenant's data doesn't
delete every copy of it:

- *The write-ahead log.* Every insert is written to the WAL before the table. With WAL archiving
  (point-in-time recovery) or streaming replicas, those segments are kept for the retention
  window, readable by anyone with access to the archive.
- *Backups.* A base backup or `pg_dump` is a snapshot. `drop_tenant` removes the partitions'
  files from the live database immediately (`test_drop_tenant_removes_its_partitions_and_nothing_else`), but every backup taken before still has them, and a
  restore, even a partial one to debug something, brings them back.
- *Dead rows.* A plain `DELETE` marks rows dead; the bytes stay in the data files until `VACUUM`
  reuses the space. Dropping a partition avoids that; deleting rows doesn't.

This matters when a customer leaves or asks for erasure: "deleted" has to mean "unrecoverable",
and with backups it doesn't, until every backup containing them has expired. The standard fix is
**crypto-shredding**: encrypt each tenant's content with its own key held in a KMS, and destroy
the key on offboarding, so every copy everywhere becomes unreadable at once. It has a real cost
here: Postgres can't run full-text search over ciphertext, so the keyword index and the
embeddings would stay plaintext, and both leak content to anyone holding a backup (lexemes
directly, embeddings through inversion attacks). The honest options are crypto-shredding plus
accepting that, or a backup retention period no longer than the contractual deletion window.

## Two things I didn't expect

**The planner can't see the tenant.** `current_setting('app.tenant_id')` isn't known at plan
time, so on a shared table Postgres plans every tenant with the same estimate: table rows ÷
distinct tenants. With a few large tenants it orders by the HNSW index for everyone, HNSW
returns its 40 nearest rows across *all* tenants, and RLS filters afterwards: a 5-chunk tenant
gets 0 of 5 results. pgvector 0.8's iterative scan fixes the recall and creates the [timing leak](#the-timing-leak-measured). Partitioning fixes both.

**Fusion only works when the scores mean something.** Two fusion methods, fixed in advance and
not tuned: reciprocal rank fusion (k=60) and relative score fusion (min-max normalise each list,
average, alpha 0.5). bge-small, hit@1:

| Keyword arm | alone | + RRF | + relative score |
| --- | --- | --- | --- |
| Postgres `ts_rank_cd` (no IDF) | 0.69 | 0.70 | 0.63 |
| BM25 (per-tenant statistics) | 0.69 | 0.83 | **0.96** |

RRF keeps only rank, so a chunk that is mediocre in both lists beats one that is first in one.
Relative fusion keeps the margin, which helps only if the margin is real: BM25 has IDF, so an
exact lease-reference match scores far above everything else; `ts_rank_cd` weighs "rent" and
"lease", which are in every chunk, as much as the reference number. RRF is also the least stable
method here: pgvector builds HNSW graphs with random level assignment, and RRF's hit@1 ranged
from 0.74 to 0.83 across index rebuilds while relative fusion was 0.96 every time.

## Retrieval numbers

`make rag-eval`: 54 questions over the demo corpus (18 per tenant, 99 chunks each), k=5,
BAAI/bge-small-en-v1.5, CPU, 2026-10-19.

| Question type | n | vector | keyword (BM25) | **hybrid (default)** |
| --- | --- | --- | --- | --- |
| Policy, paraphrased ("how much extra if rent is late?") | 15 | 1.00 | 0.87 | **1.00** |
| Unit-specific ("when does Dana Whitfield's lease run out?") | 24 | 1.00 | 0.42 | **1.00** |
| Identifier ("who is on lease SR-26-6977?") | 12 | 0.08 | 1.00 | **0.83** |
| Tenant-only document | 3 | 1.00 | 0.67 | **1.00** |
| **All, hit@1** | 54 | 0.80 | 0.69 | **0.96** |
| All, hit@5 | 54 | 0.87 | 0.93 | **0.98** |
| Cross-tenant hits, incl. probes (each tenant asked the other two tenants' document-only questions) | | 0 | 0 | **0** |

Hybrid with RRF instead: 0.74–0.83 hit@1, 0.96–0.98 hit@5 across rebuilds. p50 latency per
query: ~3 ms keyword, ~55 ms vector or hybrid, almost all of it embedding the query on CPU.

**Read these as directional.** The corpus is synthetic, small, and written by the same person who
wrote the questions. What it is good for is comparing methods on the same data, and the
cross-tenant row, which is measured on every run rather than assumed. CI runs the eval with the
hash embedder (offline, no model download) and fails on any cross-tenant hit.

## Durable runs: what a crash does at each point

What happens when the worker dies at each point in a step:

| Worker dies... | What Temporal sees | What happens on the next worker |
| --- | --- | --- |
| during the LLM call | heartbeat stops → attempt times out after 20 s | LLM call retried (attempt 2). The model is re-asked the same question; nothing was recorded, nothing is lost. |
| after the LLM call returned, before the result was recorded | activity never completed | Same as above. Temporal only trusts recorded completions. |
| during a tool execution | heartbeat stops → attempt times out after 10 s | Tool retried. Journal miss (it never finished) → tool runs again. For a non-idempotent tool this is the one case that needs the tool itself to be safe to retry, or to be journaled *before* its side effect. |
| after the tool finished and journaled, before Temporal recorded completion | activity never completed | Tool retried → **journal hit** → result returned without re-execution. This is the case the journal exists for. |
| between steps (workflow task) | workflow task times out | Replay. No activity re-runs; the workflow continues from the exact step. |

The remaining gap, the third row, is where the sandbox project comes in: journaling intent
before the side effect and compensating on duplicate, not just caching results after.

**What's still open.** The journal covers crash-after-complete. A crash *during* a non-idempotent tool, before the
journal write, still re-runs the tool on retry. Closing that needs the tool to record intent
before acting and reconcile on retry, which is the sandbox project, not this one.

## Wrangling messy tables

PR 5 loads hand-written mapping specs over a small fixture world: 5 plants named three ways, three
sources (an ERP CSV, a planner workbook with merged headers and subtotals, and a `;`-delimited
legacy report with European decimals and parentheses negatives). The expected facts were written
from the generator's ground truth, not by parsing the files back. All 190 facts load, each one
re-reads to its cell, and the one planted disagreement is the only conflict group.

Four bugs the fixtures and tests caught before anything shipped:

- **A re-export would have doubled every total.** `ingest` re-adds all facts when a document gets a
  new version, and the old version's facts stayed live. For a filing that's right: a restatement
  should conflict. For next month's export of the same sheet, `query_table` would have summed both.
  Hence `on_new_version="replace"`.
- **Replacing facts left phantom conflicts.** Deleting the old version's facts left the facts they
  had disagreed with still flagged, sometimes in a group of one. Deleting now re-checks those
  groups and dissolves any whose remaining members agree.
- **`10-07` and `100-7` normalised to the same plant.** Dropping separators before stripping leading
  zeros merges digit runs. Zeros now go first. A false merge silently sums two plants' numbers,
  which is worse than any reject.
- **`DET-07` parsed as the number −7.** The number parser stripped "any three capital letters" as a
  currency code. Only ISO 4217 codes count now.

One fixture detail: openpyxl can't write cached formula values, and "formula without a cached
value is a reject, never a 0" is the case worth testing. So `formulas.xlsx` is the one committed
binary, converted by LibreOffice and then edited to drop one cached value; its sha256 is pinned.

Limits, all deliberate for now:

- **One cell, one fact.** Line-level extracts that need summing (40 invoice lines → one
  plant-month) aren't supported. Summing needs provenance to a *set* of cells.
- **Composite entities, not dimensions.** Plant × part is the entity `Detroit Stamping 07/P-1234`.
  "All parts at Plant 7" is a `LIKE` on the canonical name, which works but is only a convention.
- **`add_facts` is two statements per fact,** so it's fine at thousands of facts and unmeasured
  beyond that.
- **Every sheet is embedded as a document** (about 700 chunks for a 10k-row sheet).
- **Crosswalks own their entities' aliases;** loading one replaces aliases another adapter set.

## Costs and limits

### Costs of this design

- One partition per tenant per table means seven tables and their indexes per customer.
  Postgres copes with thousands of partitions, and planning stays flat because the store names
  the tenant's own (measured flat from 8 to 208 tenants); tens of thousands of tiny tenants would want a different layout (a database or
  schema per cohort), and pooling small tenants into a shared partition brings the timing
  channel back between them.
- Onboarding a tenant is DDL (`create_tenant`), run by an operator connection, not the app.
- BM25's document frequencies are recomputed from the tenant's chunks on every keyword query.
  Fine at this size; a large tenant would want a maintained term-stats table.

### What's not covered

Per-tenant encryption (see [Logs and backups](#logs-and-backups)), embedding inversion if vectors leave the database,
catalog metadata (anyone who can run SQL as the app role can count the partitions, read their
row estimates and, with `pg_get_partition_constraintdef`, their tenant ids; that same role can
also just set `app.tenant_id` and read the rows, so the boundary against a SQL-capable attacker
is the role, not the catalog), and the authentication layer: RLS enforces whichever tenant the code sets, so the boundary is
only as good as the line that maps a logged-in user to `tenant_id`. The model provider sees
retrieved passages by design. Facts are loaded by code today (`add_facts`, the demo seed, the source adapters);
`rag_extraction_jobs` is there to record an LLM extraction run, but `fde rag extract` does
not exist yet.
