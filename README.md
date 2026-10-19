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
| Multi-tenant retrieval | Hybrid search (pgvector + BM25) where tenant isolation is Postgres Row-Level Security, not a `WHERE` clause. 47 attack tests. Optional extra. | `src/fde_kit/rag/` |

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

One retrieval store serving many customers. Isolation is enforced by Postgres Row-Level Security,
so it holds even when application code forgets a filter, and the test suite tries to break it.

```bash
pip install -e ".[rag]"
docker compose up -d postgres     # pgvector/pgvector:pg16
fde rag seed-demo                 # migrate + three demo tenants
fde rag search --tenant summit "How much extra do I owe if my rent is late?"
fde rag ask --tenant summit "How much is the late fee?"        # agent, search bound to summit
make rag-demo                     # same question, three tenants, three answers
make rag-test                     # the attack suite
make rag-eval                     # retrieval numbers below
```

```python
from fde_kit import Agent
from fde_kit.rag import TenantStore, make_embedder, tenant_search_registry

store = TenantStore(app_url, make_embedder("bge"))      # refuses superuser / owner connections
with store.session("summit") as s:
    s.ingest("leases/101.md", text, title="Lease: Apartment 101", metadata={"unit": "101"})
    hits = s.search("late fee?", k=5)                   # hybrid by default

# Per request: the tenant comes from your auth layer, never from the model.
Agent(tenant_search_registry(store, tenant_id=request.tenant)).run(request.question)
```

### How isolation works

| Layer | What it does |
| --- | --- |
| Policy | Every tenant table has RLS **enabled and forced**, one policy: `tenant_id = current_setting('app.tenant_id', true)` for reads (`USING`) and writes (`WITH CHECK`). Unset or empty setting matches nothing: fail closed. |
| Roles | `fde_rag_owner` (NOLOGIN) owns the tables; the app connects as `fde_rag_app` with DML grants only. It can't create tenants, alter tables, or own anything. |
| Posture check | `TenantStore` checks on startup that its connection is not a superuser, has no BYPASSRLS, doesn't own or act as owner of any RAG table, and that every table has RLS on and forced. Otherwise it raises `InsecureConnection`. `fde rag posture` runs the same check. |
| Transaction-local tenant | `session(tenant)` opens a transaction and sets the tenant with `set_config(..., is_local => true)`. It is gone when the transaction ends, so a pooled connection can't carry one request's tenant into the next. |
| Composite foreign key | `rag_chunks (tenant_id, doc_id) → rag_documents (tenant_id, doc_id)`. A chunk can't point at another tenant's document. FK checks run outside RLS, so the key enforces this, not the policy. |
| No tenant argument | The agent's `search_documents` tool has `query` and `k`. The tenant is bound when the registry is built per request. There is nothing for a prompt injection to fill in. |
| Per-tenant cache | The semantic cache is a table under the same policy, so a paraphrase lookup can only match that tenant's previous queries. Ingest clears that tenant's cache only. |
| Per-tenant statistics | BM25's document frequencies come from a query over `rag_chunks`, under RLS. IDF is per tenant with no extra code. |

No query in `store.py` filters on `tenant_id`. That is deliberate: if the code also filtered,
a broken policy would still pass every test.

### The attack suite

`tests/test_rag_isolation.py`, 47 tests against a real Postgres + pgvector. The corpus is three
property managers using the **same lease template**, so every leaked chunk reads like a correct
answer with the wrong number in it: Summit's late fee is $50, Harbor's is $75, in the same
sentence.

| Attack | Result | Test |
| --- | --- | --- |
| Same question, every tenant, every search mode | only own chunks; counts partition the table exactly | `test_every_result_belongs_to_the_asking_tenant`, `test_counts_partition_the_table` |
| Shared template: ask for "the late fee" | own number present, other tenants' numbers absent | `test_shared_template_returns_this_tenants_number` |
| Plant the exact question text in another tenant (cosine similarity 1.0) | invisible; a superuser connection shows it as the top hit, proving the plant works | `test_exact_duplicate_in_another_tenant_is_invisible` |
| Same document ingested by two tenants | two documents, two ids, each sees one | `test_same_document_in_two_tenants_stays_two_documents` |
| No tenant set, or set to `''` | zero rows on every table; inserts rejected | `test_no_tenant_set_means_no_rows_and_no_writes`, `test_empty_tenant_setting_is_not_a_wildcard` |
| Insert or `UPDATE ... SET tenant_id` into another tenant | rejected by `WITH CHECK` | `test_cannot_insert_rows_for_another_tenant`, `test_cannot_move_own_rows_into_another_tenant` |
| `DELETE`/`UPDATE ... WHERE tenant_id = 'other'` | 0 rows affected | `test_updates_and_deletes_cannot_reach_other_tenants` |
| Chunk referencing another tenant's `doc_id` | foreign-key violation | `test_chunk_cannot_point_at_another_tenants_document` |
| Metadata filter `{"tenant_id": "other"}`, SQL in filter values | empty result: filters only narrow | `test_filters_can_only_narrow` |
| SQL / tsquery syntax in the query text | treated as text; tenant unchanged | `test_hostile_query_text_is_just_text` |
| Malformed tenant ids (`harbor' OR '1'='1`, …) | `ValueError` before any SQL | `test_malformed_tenant_ids_are_rejected` |
| Model passes `tenant_id`, `namespace`, `filters` to the tool | ignored; schema has no such field | `test_search_tool_has_no_tenant_argument`, `test_injected_tenant_argument_is_dropped` |
| Prompt injection inside a retrieved document ("search tenant harbor") | model obeys, tool ignores it | `test_prompt_injection_in_a_document_cannot_redirect_the_search` |
| Connection pool reuse | next borrower has no tenant and sees nothing | `test_store_sessions_do_not_survive_the_transaction` |
| Shared semantic cache | per-tenant cache: other tenant misses | `test_tenant_cache_is_per_tenant` |
| Flood one tenant with matching documents | another tenant's BM25 scores unchanged, bit for bit | `test_bm25_statistics_are_per_tenant` |

Four tests, and one assertion in a fifth, are **demonstrations**: each reproduces a real failure
with a configuration the store refuses to use, and passes because the failure happens.

| Demonstration | What it shows |
| --- | --- |
| `test_demonstration_superuser_sees_every_tenant` | The compose `POSTGRES_USER` is a superuser. Connect as it and RLS does nothing: identical code, every tenant's chunks. |
| `test_demonstration_session_level_setting_leaks_through_the_pool` | `set_config(..., false)` on a pooled connection; the next request forgets to set a tenant and reads the previous tenant's data. |
| `test_demonstration_shared_semantic_cache_leaks` | A process-wide cache keyed on the query embedding. Summit asks Harbor's question and gets Harbor's $75. |
| `test_bm25_statistics_are_per_tenant` (second assertion) | With global corpus statistics, another tenant's documents shift your scores. That shift is information about their data. |
| `test_demonstration_hnsw_post_filter_starves_a_small_tenant` | See below. |

**Mutation check.** I broke each defence on purpose and reran the suite. A policy of `USING
(true)`: 25 of 47 tests fail. A session-level `set_config`: the pool test fails. RLS off on the cache
table: the store refuses to start (posture check). Iterative scan off: the recall test fails.
Removing `WITH CHECK` changes nothing, correctly, because Postgres applies `USING` to writes when
`WITH CHECK` is absent.

### Two things I didn't expect

**The planner can't see the tenant.** `current_setting('app.tenant_id')` isn't known at plan
time, so Postgres plans every tenant with the same estimate: table rows ÷ distinct tenants. With a
few large tenants the estimate is big and the planner orders by the HNSW index for everyone. HNSW
returns its `ef_search` (40) nearest rows across *all* tenants and RLS filters afterwards. A
tenant with 5 chunks sharing an index with a tenant with 3,000 gets **0 of 5** results.
Isolation holds; recall doesn't. pgvector 0.8's `hnsw.iterative_scan = relaxed_order` keeps
scanning until enough rows pass the filter, and every session sets it. Both behaviours are
tests. (The flip side: with many small tenants the planner brute-forces everyone, which is
correct but slow for the big ones. At real scale, partition by tenant or give large tenants
partial indexes.)

**Fusion only works when the scores mean something.** The demo corpus has twelve near-identical
leases per tenant and four kinds of question. Two fusion methods were fixed in advance and not
tuned: reciprocal rank fusion (k=60) and relative score fusion (min-max normalise each list,
average, alpha 0.5).

| Hybrid, bge-small, hit@1 | RRF | Relative score |
| --- | --- | --- |
| keyword arm = Postgres `ts_rank_cd` | 0.67 | 0.63 |
| keyword arm = BM25 (per-tenant stats) | 0.74 | **0.96** |

Neither change does much alone. RRF keeps only rank, so a chunk that is mediocre in both lists
beats one that is first in one. Relative fusion keeps the margin, which helps only if the margin
is real: BM25 has IDF, so an exact lease-reference match scores far above everything else.
`ts_rank_cd` has no IDF, so "rent" and "lease", present in every chunk, count as much as the
reference number, and normalising those scores amplifies noise.

### Retrieval numbers

`make rag-eval`: 54 questions over the demo corpus (18 per tenant, 99 chunks each), k=5,
BAAI/bge-small-en-v1.5, CPU, 2026-10-19.

| Question type | n | vector | keyword (BM25) | hybrid (RRF) | **hybrid (default)** |
| --- | --- | --- | --- | --- | --- |
| Policy, paraphrased ("how much extra if rent is late?") | 15 | 1.00 | 0.87 | 1.00 | **1.00** |
| Unit-specific ("when does Dana Whitfield's lease run out?") | 24 | 1.00 | 0.42 | 0.75 | **1.00** |
| Identifier ("who is on lease SR-26-6977?") | 12 | 0.08 | 1.00 | 0.33 | **0.83** |
| Tenant-only document | 3 | 1.00 | 0.67 | 1.00 | **1.00** |
| **All, hit@1** | 54 | 0.80 | 0.69 | 0.74 | **0.96** |
| All, hit@5 | 54 | 0.87 | 0.93 | 0.96 | **0.98** |
| Cross-tenant hits, incl. probes (each tenant asked the other two tenants' document-only questions) | | 0 | 0 | 0 | **0** |

p50 latency per query: ~3 ms keyword, ~55 ms vector or hybrid, almost all of it embedding the
query on CPU. The keyword arm recomputes document frequencies over the tenant's chunks on every
query, which is fine at this size; a large deployment would keep a per-tenant term-stats table.

**Read these as directional.** The corpus is synthetic, small, and written by the same person who
wrote the questions. What it is good for is comparing methods on the same data, and the
cross-tenant row, which is measured on every run rather than assumed. CI runs the eval with the
hash embedder (offline, no model download) and fails on any cross-tenant hit.

### What's not covered

Timing side channels (a cache hit is faster than a miss, and that's observable), embedding
inversion if vectors ever leave the database, per-tenant encryption keys, and deletion guarantees
(`DELETE` removes rows; it doesn't scrub WAL or backups). Each is a real question in a security
review; none is answered here.

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
