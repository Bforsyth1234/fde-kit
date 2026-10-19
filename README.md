# fde-kit

**v0.** A slim, opinionated starter kit for putting an LLM agent in front of a customer and
knowing whether it works. Four pieces, nothing else:

| Piece | What it is | Where |
| --- | --- | --- |
| Agent loop | ~150 lines. Prompt → LLM → tools → … → answer. Every step is a span. | `src/fde_kit/agent.py` |
| Tool registry | `@registry.tool` on a typed Python function. Schema derived from the signature. Errors returned to the model, not raised. | `src/fde_kit/tools.py` |
| Tracing | OpenTelemetry → Arize Phoenix (one container, no config). Token counts and tool I/O per span. | `src/fde_kit/tracing.py`, `docker-compose.yml` |
| Eval harness | 20-case golden set, deterministic checks + optional LLM-as-judge, pass-rate gate wired into CI. | `evals/` |
| Durable runtime | The same loop as a Temporal workflow. Kill the worker mid-run; the run finishes on the next one. Optional extra. | `src/fde_kit/durable/` |
| Multi-tenant retrieval | Hybrid search (pgvector + BM25) with tenant isolation enforced by Postgres: forced RLS, a partition per tenant, no parameter logging, redacted traces. 68 tests, including timing. Optional extra. | `src/fde_kit/rag/` |

Provider-agnostic behind a 30-line interface; Anthropic is the default, a deterministic mock is
built in for tests and offline eval runs.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d            # Phoenix UI at http://localhost:6006 (optional)

fde run "If the refund window were doubled, how many days would it be?"
fde eval                        # golden set; exit 1 if pass rate < FDE_EVAL_PASS_THRESHOLD
fde eval --provider mock        # offline harness self-check, no API key needed
pytest
```

## Add a tool

```python
from fde_kit import Agent, ToolRegistry, init_tracing

init_tracing()
registry = ToolRegistry()

@registry.tool
def shipping_estimate(weight_kg: float, express: bool = False) -> str:
    """Estimate shipping cost in USD. Express doubles the rate."""
    return f"${(4.5 + 1.2 * weight_kg) * (2 if express else 1):.2f}"

result = Agent(registry).run("Express shipping for a 3.5 kg parcel?")
print(result.output, result.tools_called, result.usage)
```

The docstring becomes the tool description the model reads; the type hints become the JSON
schema. There is no second place to keep them in sync. See `examples/demo.py`.

## Evals

`evals/golden.jsonl` has one case per line:

```json
{"id": "multi-01", "category": "multi-tool",
 "prompt": "If the refund window were doubled, how many days would it be?",
 "checks": [{"type": "tool_called", "value": "lookup_policy"}, {"type": "contains", "value": "60"}],
 "mock": [{"tool_calls": [{"name": "lookup_policy", "input": {"topic": "refund policy"}}]},
          {"tool_calls": [{"name": "calculator", "input": {"expression": "30 * 2"}}]},
          "Doubled, the refund window would be 60 days."]}
```

Check types: `exact`, `contains`, `any_of`, `regex`, `tool_called`, `no_tool`, `judge`. A case
passes only if every check passes. The report shows pass rate overall and per category, p50/p95
latency, and token totals; `--json` writes the full per-case record for diffing between model
versions.

Every case carries a `mock` script. `--provider mock` replays it, which tests the harness and the
tools rather than the model. If a case passes in mock mode and fails against the real model, the
model is the problem. If it fails in both, the check or the tool is. Keep both honest.

## Durable runs (Temporal)

The in-process loop above dies with its process. `fde_kit.durable` is the same loop as a Temporal
workflow: every LLM call and every tool call is an activity, the conversation state is workflow
state, and a run survives the worker that was executing it.

```bash
pip install -e ".[durable]"
docker compose up -d              # Temporal dev server on :7233, UI on :8233
fde worker                        # terminal 1
fde durable-run "If the refund window were doubled, how many days would it be?"   # terminal 2
```

### The crash demo

```bash
python examples/crash_demo.py             # mock provider; add --live for the real model
```

It starts a worker, starts a run whose first tool blocks for eight seconds, waits until that tool
is executing, and `SIGKILL`s the worker. The run is still `RUNNING` with no worker alive. It
then starts a second worker and prints the result and the activity history:

```
 #   activity       attempts   completed
 1   llm_call              1   yes        <- ran once, before the kill; never re-run
 2   tool_execute          2   yes        <- was executing when the worker died; retried
 3   llm_call              1   yes
 4   tool_execute          1   yes
 5   llm_call              1   yes
```

The same scenario is a test (`tests/test_durable.py::test_sigkill_worker_mid_tool_then_resume`)
that SIGKILLs a worker subprocess against a local dev server and asserts exactly that table.

### What makes it work

| Concern | Where | How |
| --- | --- | --- |
| State survives the process | `durable/workflow.py` | `messages`, `steps`, token counts are workflow variables. A new worker replays event history to rebuild them. |
| Determinism | `durable/workflow.py` vs `durable/activities.py` | The workflow body has no I/O, clocks or randomness. Both side effects are activities. |
| Fast failure detection | `durable/activities.py` | Activities heartbeat every second while running; `heartbeat_timeout` is 10 s for tools, 20 s for LLM calls, so a dead worker is noticed in seconds, not at `start_to_close_timeout`. |
| No double execution | `durable/journal.py` | Tool activities carry an idempotency key `(workflow_id, turn, tool_use_id)`. The result is journaled *before* the activity returns; a retry after a crash-between-complete-and-record is served from the journal. Postgres-backed so every worker sees every record; see below. |
| Human approval | `requires_approval=True` on a tool; `approve`/`reject` signals | The run parks on a durable timer. No worker needs to be alive while a person decides. See below. |
| Safe upgrades | `workflow.patched("v2-summarize-results")` | Old runs stay on the old path to completion; new runs take the new path. Deploy without draining. See below. |
| Visibility | `AgentWorkflow.status` query | Ask a running agent which turn, which phase, which tool, and what it's waiting on, without touching the worker. |
| Retries with backoff | `LLM_RETRY`, `TOOL_RETRY` | Rate limits and transient tool failures retry up to 5 times with exponential backoff. |

### The failure matrix

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

### Approval gates

Some tools shouldn't run without a person saying so. Mark them:

```python
@registry.tool(requires_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email on the user's behalf."""
    ...
```

or gate a tool for one run only with `fde durable-run "..." --gate send_email`.

When the model asks for a gated tool, the workflow records what it wants to do, sets its phase
to `awaiting_approval`, and waits on a Temporal timer (default 7 days). That wait is state on
the server, not a sleeping process: you can stop every worker, deploy a new version, come back
the next morning, and the run is still there, still waiting. A signal resumes it:

```bash
fde durable-run "Email ops that the incident is resolved"
# ── approval required ──────────────────────────────────────────
# turn 2: the agent wants to call send_email with
# {"to": "ops@example.com", "subject": "Incident resolved", "body": "..."}
# approve? [y/N/note]

# or, from any other terminal, laptop, or a Slack bot:
fde pending agent-3ddf081f259b
fde approve agent-3ddf081f259b toolu_01A09q90qw90lq917835lq9 --by brooks
fde reject  agent-3ddf081f259b toolu_01A09q90qw90lq917835lq9 --note "wrong recipient"
```

| Decision | What the model sees next |
| --- | --- |
| approved | The tool runs (idempotency-journaled as usual) and its result comes back as a normal `tool_result`. |
| rejected | A `tool_result` with `is_error: true` and the reviewer's note, plus an instruction not to retry. The model explains or finds another route. |
| timeout | Same as rejected, with the note "approval timed out". The run never hangs forever. |

Every decision is recorded on the step (`steps[n].approvals`) with who made it and when, so the
audit trail is part of the run's output, not a separate log.

Tests cover all three outcomes plus the one that matters:
`test_parked_run_survives_worker_restart_then_approved` parks a run, shuts down the only worker,
sends the approval with nothing serving the queue, starts a new worker, and asserts the run
completes with the right reviewer recorded.

### Upgrading a running agent

Changing the workflow code while runs are in flight is the thing durable execution makes
dangerous: a worker replaying an old history through new code that issues different commands
fails the workflow task with a non-determinism error. v2 of the workflow adds a
`summarize_result` activity for long tool outputs, which is a new command in the sequence, so
it is guarded:

```python
if workflow.patched("v2-summarize-results"):
    ... summarize, then use the summary ...     # new runs
else:
    ... use the raw result ...                  # runs that started on v1
```

`patched()` is sticky per run. A run that first evaluates it while replaying a v1 history is
pinned to the old branch for the rest of its life; a run that first evaluates it live records a
marker and stays on v2. The consequence for agents is the one you want: **a deploy never changes
the shape of an in-flight conversation.** The model's context in run A is exactly what it would
have been had you never deployed.

```bash
python examples/upgrade_demo.py
```

```
run A  (started on v1, finished on v2)          run B  (started on v2)
turn  ver  tool          chars  summarized      turn  ver  tool          chars  summarized
   1    1  fetch_report   3650  no                 1    2  fetch_report     39  yes
   2    1  wait_seconds     12  no                 2    2  wait_seconds     12  no
   3    1  fetch_report   3650  no                 3    2  fetch_report     39  yes
summarize_result calls: 0  failed tasks: 0      summarize_result calls: 2  failed tasks: 0
```

The rollout is three deploys: `patched()` (two live paths), then `deprecate_patch()` once no
v1 runs are open (one path, old histories still replay), then remove the call once v1 histories
have aged out of retention.

**The pre-deploy check.** `tests/fixtures/workflow_v1.py` is the workflow frozen at the
previous commit. `tests/test_versioning.py` starts a run on it, swaps in the live worker mid-
activity, and asserts zero failed workflow tasks, zero summarizer calls, and no patch marker.
It also replays a complete v1 history through the live code with `temporalio.worker.Replayer`,
which is the one-liner to run against production histories before any deploy that touches a
workflow:

```python
await Replayer(workflows=[AgentWorkflow]).replay_workflow(history)   # raises on drift
```

### The shared journal

The idempotency journal only helps if the worker that retries can see what the worker that
died had recorded. A file on worker A's disk is invisible to worker B. So the journal is a
Postgres table (`tool_journal`, one row per idempotency key, `INSERT ... ON CONFLICT DO
NOTHING`), and the file backend is for local development only.

```bash
docker compose up -d                 # includes postgres on :5432
FDE_JOURNAL_URL=postgresql://fde:fde@localhost:5432/fde fde worker
```

`tests/test_journal.py` reproduces the exact failure, both ways. A `charge_card` tool appends
to a file every time it really runs. The test kills worker A in the window after the tool's
result is journaled but before Temporal records the activity as complete, then lets worker B
retry it:

| Journal | Workers share it? | Card charged | `from_journal` on the result |
| --- | --- | --- | --- |
| file, one directory per worker | no | **2 times** | false |
| Postgres | yes | **1 time** | true |

The first row is kept as a test on purpose. It is the failure mode, made reproducible, that the
second row fixes. CI runs both against a Postgres service container.

The window is microseconds wide in real life, so the activity has a test-only hook
(`FDE_TEST_HANG_AFTER_JOURNAL`) that holds it open after the journal write. Never set it in
production; it exists so the crash lands in the right place every time instead of one run in
ten thousand.

### What's still open

The journal covers crash-after-complete. A crash *during* a non-idempotent tool, before the
journal write, still re-runs the tool on retry. Closing that needs the tool to record intent
before acting and reconcile on retry, which is the sandbox project, not this one.

## Multi-tenant retrieval

One retrieval store serving many customers. Isolation is enforced by Postgres, not by
application `WHERE` clauses, and the test suite tries to break it: through the data, through
timing, through the logs, and through the traces.

```bash
pip install -e ".[rag]"
docker compose up -d postgres     # pgvector/pgvector:pg16
fde rag seed-demo                 # migrate + three demo tenants
fde rag search --tenant summit "How much extra do I owe if my rent is late?"
fde rag ask --tenant summit "How much is the late fee?"        # agent, search bound to summit
fde rag posture                   # would this connection let anything leak?
make rag-demo                     # same question, three tenants, three answers
make rag-test                     # the attack suite
make rag-eval                     # retrieval numbers below
```

```python
from fde_kit import Agent
from fde_kit.rag import TenantStore, create_tenant, make_embedder, tenant_search_registry

create_tenant(admin_url, "summit")                      # operator action: row + partitions
store = TenantStore(app_url, make_embedder("bge"))      # refuses connections that could leak
with store.session("summit") as s:
    s.ingest("leases/101.md", text, title="Lease: Apartment 101", metadata={"unit": "101"})
    hits = s.search("late fee?", k=5)                   # hybrid by default

# Per request: the tenant comes from your auth layer, never from the model.
Agent(tenant_search_registry(store, tenant_id=request.tenant)).run(request.question)
```

### How isolation works

| Layer | What it does |
| --- | --- |
| Policy | Every tenant table and every partition has RLS **enabled and forced**, one policy: `tenant_id = current_setting('app.tenant_id', true)` for reads (`USING`) and writes (`WITH CHECK`). Unset or empty setting matches nothing: fail closed. |
| Partition per tenant | Documents, chunks and cache are LIST-partitioned by tenant; `create_tenant` makes the partitions. Each tenant's B-trees, GIN and HNSW indexes are physically its own, so the work a query does can't depend on anyone else's data. |
| Direct partition queries | The store names the tenant's partitions in every statement, so Postgres plans against one tenant's tables, not all of them. Each partition has its own forced policy, so a session pointed at the wrong partition still sees nothing. |
| Random ids | Document and chunk ids are UUIDv4, not a shared sequence, so the gap between two of your ids says nothing about other tenants' inserts. |
| Roles | `fde_rag_owner` (NOLOGIN) owns the tables; the app connects as `fde_rag_app` with DML grants only. It can't create tenants, alter tables, or own anything. |
| No parameter logging | `migrate` pins `log_parameter_max_length` and `log_parameter_max_length_on_error` to 0 for the app role, and every session re-pins the second one. Slow-query and error logs show `$1`, never document text or questions. |
| Posture check | `TenantStore` refuses to start unless: not a superuser, no BYPASSRLS, not an owner of any table or partition; RLS enabled and forced everywhere; the tables partitioned; **the only permissive policy on each table and partition is `tenant_isolation`, exactly as written** (permissive policies are OR-ed, so one extra `USING (true)` opens everything); parameter logging off. `fde rag posture` runs the same check. |
| Transaction-local tenant | `session(tenant)` sets the tenant with `set_config(..., is_local => true)`. It ends with the transaction, so a pooled connection can't carry one request's tenant into the next. |
| Composite foreign key | `rag_chunks (tenant_id, doc_id) → rag_documents (tenant_id, doc_id)`. A chunk can't point at another tenant's document. |
| No tenant argument | The agent's `search_documents` tool has `query` and `k`. The tenant is bound when the registry is built per request. There is nothing for a prompt injection to fill in. |
| Per-tenant cache and statistics | The semantic cache and BM25's document frequencies live in the tenant's own partitions. |
| Redacted traces | A run on a tenant-bound registry tags every span with `tenant.id` and replaces content (prompt, messages, tool I/O, answer) with `[redacted: N chars]` before it is set on the span. |

No query in `store.py` filters on `tenant_id`. If the code also filtered, a broken policy would
pass every test.

### The attack suite

`tests/test_rag_isolation.py` (61 tests) and `tests/test_tracing_tenants.py` (7), against a real
Postgres + pgvector. The corpus is three property managers using the **same lease template**, so
every leaked chunk reads like a correct answer with the wrong number in it: Summit's late fee is
$50, Harbor's is $75, in the same sentence.

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
| Malformed tenant ids (`harbor' OR '1'='1`, …) | `ValueError` before any SQL | `test_malformed_tenant_ids_are_rejected` |
| Model passes `tenant_id`, `namespace`, `filters` to the tool | ignored; schema has no such field | `test_search_tool_has_no_tenant_argument`, `test_injected_tenant_argument_is_dropped` |
| Prompt injection inside a retrieved document ("search tenant harbor") | model obeys, tool ignores it | `test_prompt_injection_in_a_document_cannot_redirect_the_search` |
| Connection pool reuse | next borrower has no tenant and sees nothing | `test_store_sessions_do_not_survive_the_transaction` |
| Shared semantic cache | per-tenant cache: other tenant misses | `test_tenant_cache_is_per_tenant` |
| Flood one tenant with matching documents | another tenant's BM25 scores unchanged, bit for bit | `test_bm25_statistics_are_per_tenant` |
| **Timing**: vary tenant B's data, measure tenant A's query work | identical pages touched for vector, keyword and cache lookup, under both the default and at-scale plans | `test_query_work_does_not_depend_on_other_tenants_data` |
| **Timing**: count tenants via planning time | every plan touches only the asking tenant's partitions | `test_plans_touch_only_the_tenants_own_partitions` |
| **Ids**: count other tenants' inserts from id gaps | ids are random UUIDv4 | `test_ids_are_random_not_a_shared_counter` |
| **Logs**: parameters in slow-query / error logs | pinned to 0; the app can't re-enable the superuser-only one; sessions re-pin the other | `test_app_role_never_logs_query_parameters`, `test_store_sessions_pin_error_parameter_logging_off` |
| **Traces**: tenant data in the trace store | no span attribute contains the question, passage or answer; every span tagged | `test_tracing_tenants.py` |
| Misconfiguration: extra permissive policy, weakened partition policy, unforced RLS, parameter logging on | store refuses to start | `test_posture_check_*` |

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

### The timing leak, measured

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

### Logs and backups

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

### Two things I didn't expect

**The planner can't see the tenant.** `current_setting('app.tenant_id')` isn't known at plan
time, so on a shared table Postgres plans every tenant with the same estimate: table rows ÷
distinct tenants. With a few large tenants it orders by the HNSW index for everyone, HNSW
returns its 40 nearest rows across *all* tenants, and RLS filters afterwards: a 5-chunk tenant
gets 0 of 5 results. pgvector 0.8's iterative scan fixes the recall and creates the timing leak
above. Partitioning fixes both.

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

### Retrieval numbers

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

### Costs of this design

- One partition per tenant per table means three tables and their indexes per customer.
  Postgres copes with thousands of partitions, and planning stays flat because the store names
  the tenant's own (measured flat from 8 to 208 tenants); tens of thousands of tiny tenants would want a different layout (a database or
  schema per cohort), and pooling small tenants into a shared partition brings the timing
  channel back between them.
- Onboarding a tenant is DDL (`create_tenant`), run by an operator connection, not the app.
- BM25's document frequencies are recomputed from the tenant's chunks on every keyword query.
  Fine at this size; a large tenant would want a maintained term-stats table.

### What's not covered

Per-tenant encryption (see Logs and backups), embedding inversion if vectors leave the database,
and the authentication layer: RLS enforces whichever tenant the code sets, so the boundary is
only as good as the line that maps a logged-in user to `tenant_id`. The model provider sees
retrieved passages by design.

## Baseline results

Golden set against `claude-sonnet-5-5`, single run, 2026-10-18:

| Metric | Value |
| --- | --- |
| Pass rate | **19 / 20 (95%)** |
| By category | calculator 8/8 · lookup 5/5 · lookup-miss 1/1 · date 1/1 · date+calc 1/1 · multi-tool 2/2 · no-tool 1/2 |
| Latency | p50 2.8 s · p95 8.9 s per case (2–3 round-trips each) |
| Tokens | ~30k in / ~1.9k out for the full set (tool schemas are re-sent every turn) |

**Known failure — `notool-01`.** Asked for a one-sentence definition, the model returns two
well-formed sentences, reproducibly. The judge fails it on the length constraint while noting the
content is correct. Kept as-is: it is a real instruction-following miss and a good example of what
the judge catches that `contains` can't.

**What the first live run found.** The mock suite was green; the first real run surfaced three
bugs, all at the provider boundary: an SDK parameter that no longer exists, `.env` values not
reaching the SDK, and a thousands-separator the `contains` check didn't normalise. None were
visible offline. `tests/test_live.py` now makes one real call when a key is present so CI catches
that class of bug before the full eval does.

## CI

`.github/workflows/ci.yml` runs lint, tests (Postgres + pgvector service container), the
offline retrieval eval with its cross-tenant gate, and the offline agent eval on every push and
PR. On pushes
to `main` it also runs the golden set against the real model **if** an `ANTHROPIC_API_KEY`
repository secret is set, and fails the build below the threshold. The results JSON is uploaded
as a build artifact.

## Configuration

Everything is an environment variable with an `FDE_` prefix (or a line in `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `FDE_PROVIDER` | `anthropic` | `anthropic` or `mock` |
| `FDE_MODEL` | `claude-sonnet-5-5` | Model id |
| `FDE_MAX_TURNS` | `10` | Hard cap on LLM round-trips per run |
| `FDE_OTEL_ENABLED` | `true` | Export spans |
| `FDE_OTEL_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP/HTTP collector |
| `FDE_OTEL_TENANT_CONTENT` | `false` | Record prompts and tool I/O on spans for tenant-bound runs. Only with a per-tenant access-controlled trace store. |
| `FDE_EVAL_PASS_THRESHOLD` | `0.85` | CI gate |
| `FDE_RAG_URL` | derived from admin URL | App connection for retrieval, as `fde_rag_app`. Never a superuser. |
| `FDE_RAG_ADMIN_URL` | `postgresql://fde:fde@localhost:5432/fde` | Used only by `fde rag migrate` / `seed-demo` |
| `FDE_RAG_EMBEDDER` | `bge` | `bge` (bge-small via fastembed) or `hash` (offline, tests) |

## Design notes

- **The loop is deliberately plain.** No planner, no memory, no streaming. Each `_step` is the
  unit that a durable-execution runtime (Temporal, Step Functions) would wrap as an activity; that
  is the next project built on this kit, and the loop's shape is chosen to make it a drop-in.
- **Tool errors go back to the model.** A failing tool returns `is_error: true` with the exception
  text, and the model gets a chance to recover. The run never crashes because an integration did.
- **`max_turns` is a result, not an exception.** A runaway loop stops and reports
  `stopped_reason="max_turns"` so evals can count it.
- **Spans use OpenInference attribute names** (`llm.token_count.*`, `tool.name`, `input.value`,
  `output.value`) so Phoenix renders them without extra setup. Any OTLP backend works.
- **Mock scripts live next to the cases they test.** Offline runs are a harness self-check, and
  they are labelled as such; they are never presented as a model score.

## What's not here yet

Prompt versioning, pairwise/A-B eval mode, a customers/ onboarding directory, and a
one-command cloud deploy. Those are v1.

## License

MIT
