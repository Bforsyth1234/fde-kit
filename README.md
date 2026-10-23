# fde-kit

**v1.** A slim, opinionated starter kit for putting an LLM agent in front of a customer and
knowing whether it works, with every answer traceable to the records and the prompt behind it.

**Core** (installed by `pip install fde-kit`):

| Piece | What it is | Where |
| --- | --- | --- |
| Agent loop | ~200 lines. Prompt → LLM → tools → … → answer. Every step is a span. Numbers every source a tool returns and resolves the `[n]` citations in the answer back to them. | `src/fde_kit/agent.py`, `citations.py` |
| Tool registry | `@registry.tool` on a typed Python function. Schema derived from the signature. Errors returned to the model, not raised. A tool may return `ToolResult(content, sources=...)` to say what records its content came from. | `src/fde_kit/tools.py` |
| Prompts | Markdown files, Git-tracked, hashed. Every run records `prompt_name` and `prompt_sha256` on the result, the spans and the eval JSON. | `src/fde_kit/prompts/` |
| Tracing | OpenTelemetry → Arize Phoenix (one container, no config). Token counts, tool I/O, source ids and prompt hash per span. | `src/fde_kit/tracing.py`, `docker-compose.yml` |
| Run records | A run as a JSONL file: replay it exactly, replay with a live model, or fork it at a turn. Durable runs export to the same file. | `src/fde_kit/record.py` |
| Eval harness | 25-case golden set, deterministic checks (incl. `cites` / `cites_only`, replayed runs) + optional LLM-as-judge, pass-rate and citation-precision gates wired into CI. | `evals/` |

**Optional extras** (`pip install "fde-kit[durable]"`, `"fde-kit[rag]"`):

| Piece | What it is | Where |
| --- | --- | --- |
| Durable runtime | The same loop as a Temporal workflow. Kill the worker mid-run; the run finishes on the next one. | `src/fde_kit/durable/` |
| Multi-tenant retrieval | Hybrid search (pgvector + BM25) over versioned documents, plus a fact table with provenance and precision-aware conflicts, with tenant isolation enforced by Postgres: forced RLS, a partition per tenant, no parameter logging, redacted traces. Three cited tools: `search_documents`, `fetch_document`, `query_table` (read-only SQL). Source adapters load SEC EDGAR filings and local files. 100+ tests, including timing. | `src/fde_kit/rag/`, `rag/adapters/` |

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

## Citations

A citation is produced by the tool layer and checked by the loop. The model's prose is never
the source of truth for what was cited.

```python
from fde_kit import Source, ToolResult

@registry.tool
def lookup_policy(topic: str) -> ToolResult:
    """Look up a policy. Returns a numbered passage; cite its number."""
    text = kb[topic]
    src = Source(source_id=f"policy:{topic}", uri=f"policies/{topic}.md", title=topic.title(),
                 content_hash=hashlib.sha256(text.encode()).hexdigest())
    return ToolResult(f"[#1] {src.title} ({src.uri})\n{text}", sources=(src,))

result = Agent(registry).run("What is the uptime SLA? Cite your source.")
result.output                 # "The uptime SLA is 99.9% monthly [1]."
result.sources                # every Source any tool returned, in citation order
result.cited                  # the subset the answer cites by number: [Source(policy:sla, ...)]
result.fabricated_citations   # numbers in the answer no tool returned: [] or there is a bug
```

How it works: a tool marks its own passages `[#1]`, `[#2]`, … (local to that call). The loop
keeps one ledger per run, gives each new `Source` the next run-wide number, rewrites the local
markers before the content reaches the model, and records `citations: [n, ...]` on every tool
result in `result.steps`. The packaged system prompt asks the model to cite by `[n]`.
`result.cited` parses the answer's `[n]` tokens and resolves them through the ledger; a number
the ledger does not have is a fabricated citation, which the `cites_only` eval check fails and
`fde run` prints in red.

Why a ledger and not the model's own numbering: two searches in one run both return a `[1]`.
The tool cannot know what the run has already seen; the loop can. Why `Source.content_hash`:
the citation can prove what text it pointed at after the document is re-ingested. `fde rag ask`
prints the table of sources seen and which the answer cited.

## Prompt versioning

The system prompt is a Markdown file, and every run records which one it used.

```
prompts/default.md             # your project's prompt, if the directory exists
src/fde_kit/prompts/default.md # packaged fallback
```

`FDE_PROMPT=<name>` picks `prompts/<name>.md` (or a path to any `.md`); `FDE_SYSTEM_PROMPT` or
`Agent(system_prompt=...)` passes raw text, recorded as `inline`. In all cases the run carries
`prompt_name` and `prompt_sha256`: on `AgentResult`, on the `agent.run` and `llm.call` spans,
in each case of `fde eval --json` and in the report header, and in a durable run's workflow
history (`AgentRunInput` / `AgentRunOutput`). Two eval files with different hashes are two
experiments; Git is the version store and the hash is how a trace points at a commit of the
prompt without needing Git at runtime.

## Record, replay, fork

A finished run becomes one file, `runs/<run_id>.jsonl`, with every LLM response and every tool
result in the order they happened. The file can be replayed without the model or the database,
replayed against a live model with the evidence frozen, or forked at a turn with a new
instruction.

| Mode | What is frozen | What re-runs | What the diff tells you |
| --- | --- | --- | --- |
| Exact replay | prompt, LLM responses, tool results | nothing | Nothing: output is byte-identical. Proves the evidence path and the citations are reconstructible from the file alone. |
| Live-LLM replay | prompt (pinned from the header), tool results | every LLM call | How the model's reasoning and wording changed on the same evidence; the `cites` checks say whether the citations held. |
| Fork | turns 1..N exactly as recorded | turn N+1 onward, with an injected instruction, live model, real tools | What a different instruction does from the same starting state. The parent file is never written to; lineage is in the child's header. |

```
fde run "prompt"                         # as today; + "run_id r-… saved to runs/r-….jsonl" if FDE_RECORD_RUNS
fde run --record PATH "prompt"            # record to PATH
fde run --replay PATH                     # exact replay; prompt taken from the record
fde run --replay PATH --live-llm          # tools from record, model live, prompt pinned from header
fde run --fork PATH --at N "instruction"  # new run; header.parent_run_id/fork_at set; written to runs/
fde rag ask --tenant T "q"                # always records to runs/ (prints the path)
fde runs list [--dir runs]                # table: run_id, created_at, runtime, tenant, model, prompt, turns, stopped, parent
fde runs tree RUN_ID                      # lineage: parents up, forks down (scans the dir)
fde runs show RUN_ID                      # header + one line per row (turn, kind, name/stop_reason)
fde durable-export WORKFLOW_ID [--out PATH]   # a Temporal run's history, as the same file
```

`--replay` prints the recorded output, then the replayed output, then `identical` or `differs`
with the first differing line; for `--live-llm` that line is the point. A replay that asks for a
tool call the record does not have next prints `replay diverged at turn N: expected
<name>(input) got <name>(input)` and exits 3.

```python
from fde_kit import Agent
from fde_kit.record import (RunRecord, fork_session, live_llm_session, recorded_prompt,
                            replay_session)

result = Agent(registry).run("What is the uptime SLA?", record="runs/sla.jsonl")
result.run_id, result.record_path            # "r-3f9a1c2e7b4d", "runs/sla.jsonl"

rec = RunRecord.load("runs/sla.jsonl")       # validates the header and the format version
rec.header["model"], rec.result["output"]    # rec.llm_rows(), rec.tool_rows(), rec.user_rows()
prompt = recorded_prompt(rec)                # the opening user row

again = Agent(registry).run(prompt, session=replay_session(rec, registry))
again.output == rec.result["output"]         # True, byte for byte
again.stopped_reason                         # "end_turn", or "replay_diverged" with diverged_at set

live = Agent(registry).run(prompt, session=live_llm_session(rec, registry, provider))

child = Agent(registry).run(
    prompt,
    session=fork_session(rec, registry, provider, at=1, instruction="Quote the clause verbatim."),
    record="runs/sla-fork.jsonl",            # child file; header carries parent_run_id and fork_at
)
```

`provider` is any `LLMProvider` (the Anthropic one, or `MockProvider` in tests). The second
argument to each session helper is the registry the run was recorded with: the model must see
the same tool schemas, and after a fork's `at` the real tools in it run.

The file, one JSON object per line, `kind` first (trimmed):

```jsonc
{"kind":"header","version":1,"run_id":"r-3f9a1c2e7b4d","created_at":"2026-10-22T23:40:11Z",
 "runtime":"agent","tenant_id":null,"model":"claude-sonnet-5-5","prompt_name":"default",
 "prompt_sha256":"9abf…","prompt_text":"…","tools":["search_documents"],"parent_run_id":null,"fork_at":null}
{"kind":"user","turn":1,"text":"How much is the late fee?","actor":"user","by":""}
{"kind":"llm","turn":1,"text":"","stop_reason":"tool_use",
 "tool_calls":[{"id":"toolu_01","name":"search_documents","input":{"query":"late fee"}}],"raw_content":[…],"usage":{…}}
{"kind":"tool","turn":1,"tool_use_id":"toolu_01","name":"search_documents","input":{"query":"late fee"},
 "content":"[#1] Lease 101 (…)\n…","is_error":false,"sources":[{"source_id":"…","content_hash":"…"}],"approval":null}
{"kind":"llm","turn":2,"text":"The late fee is $50 [1].","stop_reason":"end_turn","tool_calls":[],…}
{"kind":"result","output":"The late fee is $50 [1].","stopped_reason":"end_turn","turns":2,
 "total_ms":2412,"sources":[…],"cited":["lease:101"],"fabricated_citations":[]}
```

Two details make replay exact. A `tool` row stores the tool's raw content, with its local
`[#i]` markers, before the citation ledger rewrote them to run-wide `[n]`; a replay pushes that
content through the same ledger, so the numbering, `result.sources` and `result.cited` are
re-derived rather than copied. And a replay matches each requested tool call to the next
unconsumed `tool` row by `(name, input)`, never by `tool_use_id`: ids differ across providers
and across live reruns, while a model that asks the same tool the same question is on the same
evidence path. Anything else raises `ReplayDiverged`, which the loop reports as
`stopped_reason="replay_diverged"` with `diverged_at` and the expected and actual calls.
`llm` rows are indexed by the number of assistant messages already in the conversation, the
same way `MockProvider` is, so a fork's injected user message does not shift them.

Determinism, stated plainly: pure replay is exact. Live-LLM replay and forks reproduce the
evidence path exactly and the model's wording approximately. The `cites` checks assert the
former.

**`runs/` is local evidence.** The file is unredacted: prompt text, every passage a tool
returned, the answer. Treat it like a trace with `FDE_OTEL_TENANT_CONTENT=1`. The directory is
gitignored; delete it on your own schedule. `fde run` records only when `FDE_RECORD_RUNS=1` or
`--record` is given; `fde rag ask` always records and prints the path.

A durable run exports to the same file. `fde durable-export WORKFLOW_ID` walks the workflow's
event history: the start event gives the header and the `user` row, each `llm_call` and
`tool_execute` activity gives an `llm` or `tool` row (with the approval decision on the tool row
when there was one), and the completion gives the `result` row. The summarizer activity is
skipped so the record keeps the raw tool content; a summarized step carries
`summarized_from_chars` instead. `tests/test_durable.py::test_export_then_replay` runs the
standard mock scenario on a worker, exports it, loads it with `RunRecord.load` and replays it
through the in-process `Agent`, asserting the same output and `stopped_reason == "end_turn"`:
both runtimes, one record.

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

Check types: `exact`, `contains`, `any_of`, `regex`, `tool_called`, `no_tool`, `cites`,
`cites_only`, `cites_any`, `stopped`, `conflict_reported`, `judge`. A case passes only if every
check passes. `cites <id>` passes when the answer cites, by `[n]`, a Source whose `source_id`,
`doc_id` or `uri` is `<id>`; naming the document in prose does not count. `cites_only` passes
when no citation in the answer is fabricated. `stopped <reason>` compares
`result.stopped_reason` (`end_turn`, `max_turns`, `replay_diverged`). `conflict_reported
[values]` passes when the output contains every listed value and one of the words "conflict",
"conflicts", "disagree", "differs", "restated". The report shows pass rate overall and per category, p50/p95 latency, token
totals and, when any citation was fabricated, citation precision in red; the header carries
the model and the prompt hash. `--json` writes the full per-case record, including
`prompt_sha256`, `sources`, `cited` and `fabricated_citations`, for diffing between model or
prompt versions.

Every case carries a `mock` script. `--provider mock` replays it, which tests the harness and the
tools rather than the model. If a case passes in mock mode and fails against the real model, the
model is the problem. If it fails in both, the check or the tool is. Keep both honest.

A case may carry `"replay": "evals/runs/<file>.jsonl"` instead of `mock`: the harness replays
that record (see Record, replay, fork) and applies the checks to the result, in both providers.
The three `replay` cases cover a cited run, a two-tool run, and a record edited so the second
tool input differs, which must end with `stopped: replay_diverged`. The fixture records are
generated by `python -m evals.make_replay_fixtures` and committed; `tests/test_evals.py`
asserts they regenerate byte-identical, so a drift in the file format is caught.

`fde eval --set rag` runs a second set, `evals/golden_rag.jsonl`: 12 cases over the demo lease
corpus, each bound to a tenant, with the three retrieval tools running for real against
Postgres while the model is the mock or the live one. Four look a term up in a lease (`cites`
the lease), four reach a number through `query_table` (`tool_called`, `contains`, `cites` the
fact's document), two hit the late-fee conflict (`conflict_reported` both values, `cites` both
documents), and two ask one tenant for another tenant's unit (`cites_only`; the answer says not
found). CI seeds the demo and runs it in mock mode with a pass threshold of 1.0.

`fde eval --set edgar` runs a third set, `evals/golden_edgar.jsonl`: 12 cases over the pinned
EDGAR corpus (see Source adapters and SEC EDGAR), seeded offline from the committed fixtures.
Three `table-fact` cases reach one number through `query_table` (Moderna FY2021 revenue,
Gilead FY2021 net income, Regeneron FY2021 revenue) and cite the filing. Three
`table-conflict` cases hit a real restatement (Pfizer FY2019 revenue, Gilead FY2019 R&D,
Regeneron FY2019 revenue): `conflict_reported` every value, `cites` every filing. Two
`precision` cases (Vertex FY2019 revenue, Moderna FY2020 net loss) assert the opposite: no
`conflict=true` in the tool output, one value in the answer, no conflict wording. Two
`doc-lookup` cases answer from the first 40,000 characters of a filing, and two `cross-tenant`
cases ask `pharma-b` for Pfizer and `pharma-a` for Moderna and expect not found. CI runs it in
mock mode with a pass threshold of 1.0; no network, no SEC contact.

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
fde rag ask --tenant summit "How much is the late fee?"        # agent with the three tools, bound to summit
fde rag query --tenant summit "SELECT field, value_num, unit FROM rag_facts"   # read-only SQL under RLS
fde rag fetch --tenant summit DOC_ID [--start 0 --end 6000]   # a slice of one stored document body
fde rag dictionary --tenant summit                            # the field definitions query_table sees
fde rag posture                   # would this connection let anything leak?
fde rag seed-edgar [--offline]    # pharma-a / pharma-b from the pinned EDGAR corpus
fde rag ingest --tenant T --adapter edgar [--ticker PFE] [--offline]   # some or all pinned filers
fde rag ingest --tenant T --adapter local_files --dir PATH [--entity NAME]
fde rag edgar-drift               # live re-fetch of the pinned filings; exit 1 if SEC's copy changed
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
| Partition per tenant | Documents, chunks, cache, entities, facts, dictionary and extraction jobs are LIST-partitioned by tenant; `create_tenant` makes the partitions. Each tenant's B-trees, GIN and HNSW indexes are physically its own, so the work a query does can't depend on anyone else's data. |
| Direct partition queries | The store names the tenant's partitions in every statement, so Postgres plans against one tenant's tables, not all of them. Each partition has its own forced policy, so a session pointed at the wrong partition still sees nothing. |
| Random ids | Document and chunk ids are UUIDv4, not a shared sequence, so the gap between two of your ids says nothing about other tenants' inserts. |
| Roles | `fde_rag_owner` (NOLOGIN) owns the tables; the app connects as `fde_rag_app` with DML grants only. It can't create tenants, alter tables, or own anything. |
| Hidden tenant ids | Partition names are a hash of the tenant id; with `FDE_RAG_PARTITION_KEY` set they are an HMAC, so the catalog and the server log don't reveal tenant ids to anyone holding a candidate list. Partitions carry no comments. The key is fixed at first `migrate` (a fingerprint is stored, and a mismatched key is refused). |
| No parameter logging | `migrate` pins `log_parameter_max_length` and `log_parameter_max_length_on_error` to 0 for the app role, and every session re-pins the second one. Slow-query and error logs show `$1`, never document text or questions. |
| Posture check | `TenantStore` refuses to start unless: not a superuser, no BYPASSRLS, not an owner of any table or partition; RLS enabled and forced everywhere; the tables partitioned; **the only permissive policy on each table and partition is `tenant_isolation`, exactly as written** (permissive policies are OR-ed, so one extra `USING (true)` opens everything); parameter logging off; the partition key matches the one the schema was migrated with. `fde rag posture` runs the same check. |
| Transaction-local tenant | `session(tenant)` sets the tenant with `set_config(..., is_local => true)`. It ends with the transaction, so a pooled connection can't carry one request's tenant into the next. |
| Composite foreign key | `rag_chunks (tenant_id, doc_id) → rag_documents (tenant_id, doc_id)`, and the same shape from `rag_facts` to entities, documents, chunks and extraction jobs. A chunk or a fact can't point at another tenant's document. |
| No tenant argument | The agent's tools take `query` and `k`, a `doc_id` and a char range, or `sql`. The tenant is bound when the registry is built per request. There is nothing for a prompt injection to fill in. |
| Per-tenant cache, statistics & journal | The semantic cache and BM25's document frequencies live in the tenant's own partitions; durable tool-journal keys are scoped by `tenant_id` and `tool_name`. |
| Redacted traces | A run on a tenant-bound registry tags every span with `tenant.id`, replaces content (prompt, messages, tool I/O, answer) with a constant `[redacted]` marker (no character-length hint), and scrubs `exception.message`, `exception.stacktrace` and error status descriptions on span events before export. |

No query in `store.py` filters on `tenant_id`. If the code also filtered, a broken policy would
pass every test.

### Facts, versions and query_table

Search finds prose. Three more things are needed before an agent's answer can be checked
against the record: proof of what text a citation pointed at, after the source has changed; a
way to reach the numbers in a corpus without reading every document; and a rule for two sources
that disagree. The schema grows by five columns, two offsets and four tables, all under the
same policy.

| Table | Key columns | Purpose |
| --- | --- | --- |
| `rag_documents` (added) | `content_hash`, `version`, `fetched_at`, `superseded_by`, `body` | A source is versioned. `body` is the full text `fetch_document` slices. `UNIQUE (tenant_id, source)` becomes `UNIQUE (tenant_id, source, version)`; `migrate` finds the old constraint by its definition and swaps it. |
| `rag_chunks` (added) | `char_start`, `char_end` | Where the chunk's body sits inside `rag_documents.body`. NULL when it can't be located verbatim. |
| `rag_entities` | `entity_id`, `kind`, `canonical_name`, `aliases` | What facts are about: a property manager, a unit, a company. `UNIQUE (tenant_id, kind, canonical_name)`; names and aliases resolve case-insensitively. |
| `rag_facts` | `fact_id`, `entity_id`, `field`, `period_start`, `period_end`, `value_num` or `value_text` (exactly one, by `CHECK`), `unit`, `doc_id`, `chunk_id`, `conflict_group_id`, `extraction_job_id` | One harmonised value with its provenance. Composite foreign keys to entities (`CASCADE`), documents, chunks and jobs (`ON DELETE SET NULL (doc_id)` and so on: deleting a cited document clears that one column, never `tenant_id`). |
| `rag_dictionary` | `field`, `definition`, `unit`, `allowed_values`, `notes` | What each field means. Appended to the `query_table` tool description, so the model reads the customer's definitions, not a guess. |
| `rag_extraction_jobs` | `extraction_job_id`, `kind`, `model`, `prompt_sha256`, `started_at` | Which run wrote a fact. The demo seed is `kind='seed'`; an LLM job would carry its model and prompt hash. |

All four new tables are `PARTITION BY LIST (tenant_id)` and listed in `TENANT_TABLES` and
`PARTITIONED_TABLES`, so `migrate` applies the forced policy and the grants, `create_tenant`
makes their partitions, `drop_tenant` drops them and `check_posture` audits them, with no new
code in any of the four.

**Versioning.** `ingest(source, text)` is idempotent by `sha256(text)`; `ingest_versioned`
returns the `IngestOutcome` (`doc_id`, `version`, `changed`, `content_hash`), which `ingest`
also leaves on `session.last_ingest`.

- Same source, same hash: nothing is written; the current `doc_id` comes back with
  `changed=False`.
- Same source, new hash: a new row at `version = max + 1` with the new body, the previous
  current row's `superseded_by` pointed at it, the previous version's chunks deleted and the
  tenant's cache cleared. Search only ever sees current chunks. The old row stays, with its
  source, hash, version and body, so an old citation's `doc_id` + `content_hash` still
  resolves; its chunk ids do not.
- `fetch(doc_id)` on a superseded document returns its old body with `superseded_by` set, so a
  reader can see what changed. An unknown `doc_id` raises `KeyError`, and under RLS another
  tenant's `doc_id` is unknown, so the error is identical for "no such document" and "not yours".

**Three tools.** `tenant_search_registry(store, tenant_id, registry=None, *, cached=True,
tools=("search_documents", "fetch_document", "query_table"))` registers up to three;
`tools=("search_documents",)` is exactly the PR 1 registry. None takes a tenant.

| Tool | Returns | Each `Source` carries |
| --- | --- | --- |
| `search_documents(query: str, k: int = 5)` | passages numbered `[#i]` | `source_id` = chunk id, `uri` = document source, `doc_id`, the chunk's `content_hash` |
| `fetch_document(doc_id: str, char_start: int = 0, char_end: int = 6000)` | `[#1] {title} ({source}) v{version} chars {start}-{end} of {total}`, the text, and `next: <offset>` while more remains | `source_id` = `{doc_id}:{start}-{end}`, `doc_id`, the document's `content_hash`, `span` = `chars:{start}-{end}`. Two slices of one document are two citations, each provable. Unknown id: `No document with that id.` as an error result. |
| `query_table(sql: str)` | a header line, then one row per line. A row with a `fact_id` column gets `[#i]` and a Source; a row whose `conflict_group_id` is set gets ` conflict=true`; `SELECT count(*)` rows get neither. More than 200 rows: `... (truncated at 200 rows; add a WHERE or LIMIT)` | `source_id` = `fact_id`, `span` = `fact:{fact_id}`, `uri` from a `doc_source` or `source` column (else `fact:{fact_id}`), `doc_id`, `content_hash` from `doc_hash` or `content_hash`, title from `entity` + `field`. The model's SELECT decides which of these are present. |

The `query_table` description is built once per registry: the docstring, then a paragraph
naming `rag_facts`, `rag_entities` and `rag_documents` with their columns, then one
`- field (unit): definition` line per `rag_dictionary` row. One query per request, not per call.

**`query_table` has three layers, and only two of them are the boundary.**

1. *The guard* (`rag/sqlguard.py`, `check_select`, tested without Postgres). It strips comments
   and one trailing `;`, requires the statement to start with `SELECT` or `WITH ... SELECT`,
   and rejects a second `;`, `$$`, `SELECT ... INTO`, any identifier starting with `pg_`, and
   the whole words `insert update delete merge truncate drop alter create grant revoke copy
   vacuum analyze set reset show listen notify do call execute prepare lock security
   information_schema current_setting set_config`. It is a keyword filter and says so. It is
   not aware of string literals: `SELECT * FROM rag_facts WHERE value_text = 'do not call'` is
   rejected because of `do`, and that is left as is, since too strict is the safe side.
   `deleted_at` or `reset_count` as column names are fine; whole words only.
2. *The transaction.* The statement runs after `SET LOCAL transaction_read_only = on`, under a
   statement timeout (5 s by default), as the app role, inside a tenant session whose forced
   policy filters every table the SQL can name. The model writes against the parent tables
   (`rag_facts`, `rag_entities`, `rag_documents`, `rag_dictionary`); the policy applies through
   them. The SQL goes to Postgres without psycopg parameters, so a `LIKE '%rent%'` is not
   mistaken for a placeholder. Postgres allows switching a transaction to read-only and never
   back, so after one `query()` the rest of that session is read-only; the tool opens a session
   per call, and Python code that wants to write afterwards opens a fresh one. A statement that
   slips past the filter still cannot write and still cannot see another tenant's rows.
3. *The savepoint.* The statement runs inside a savepoint, so a rejected one (a bad column, a
   timeout, a read-only violation) rolls back to it and the session continues. Any Postgres
   error comes back as `SqlRejected` with the first line of the server's message, which the
   tool returns as `Query rejected: ...` with `is_error=True`, and the run goes on.

**Conflicts are kept, not averaged.** `add_facts` inserts every row. When an existing fact has
the same `(entity_id, field, period_start, period_end)` and a different value (compared at the
facts' precision since PR 4; see below), all of them and
the new row share one `conflict_group_id` (the existing group's, else a new one). The same
value from a second source is inserted too, with no group: two sources agreeing is evidence,
not a duplicate. The demo corpus has one deliberate conflict per tenant: the lease's late fee
and a 2026 addendum's revised fee (Harbor $75 → $85, Summit $50 → $60, Oakline 5% → 6%) are
loaded on the same `(property manager, late_fee, 2026-01-01..)` key, so `query_table` returns
both rows marked `conflict=true`, the tool description tells the model to report both values
and say they conflict, and the `conflict_reported` eval check verifies that it did.

```python
from datetime import date
from fde_kit import Agent
from fde_kit.rag import DictEntry, FactRow, tenant_search_registry

with store.session("summit") as s:
    doc_id = s.ingest("summit/leases/101.md", text, title="Lease: Apartment 101")
    s.upsert_entity("property_manager", "Summit Rentals", aliases=["summit"])
    s.set_dictionary([DictEntry("late_fee", "Fee charged after the grace period.", "USD")])
    s.add_facts([FactRow("summit", "late_fee", 50, "USD",
                         period_start=date(2026, 1, 1), doc_id=doc_id)])
    s.add_facts([FactRow("summit", "late_fee", 60, "USD",
                         period_start=date(2026, 1, 1), doc_id=addendum_id)])   # conflict group
    r = s.query("SELECT fact_id, field, value_num, conflict_group_id FROM rag_facts")
    r.columns, r.rows, r.truncated                   # the rest of this session is read-only

registry = tenant_search_registry(store, tenant_id=request.tenant,
                                  tools=("search_documents", "fetch_document", "query_table"))
Agent(registry).run("What is the late fee, and does any document disagree?")
```

### Source adapters and SEC EDGAR

An adapter turns an external source into documents and facts with provenance; the store does
the rest (versioning, chunking, embedding, conflict groups, RLS). Two ship: `edgar`, for SEC
10-K filings and their XBRL financials, and `local_files`, for a directory a tenant supplies.

```python
# rag/adapters/base.py
@dataclass
class AdapterBatch:                 # one unit of ingest, e.g. one company
    entities: list[RawEntity]       # kind, canonical_name, aliases
    documents: list[RawDocument]    # source (stable URI), title, text, metadata
    facts: list[RawFact]            # document_source + a FactRow with doc_id left None
    dictionary: list[DictEntry] = field(default_factory=list)

class Adapter(Protocol):
    name: str                       # "edgar", "local_files"
    def batches(self) -> Iterator[AdapterBatch]: ...

def ingest(session: TenantSession, adapter: Adapter, *, max_chars: int = 900) -> IngestReport
```

`ingest` upserts the entities, runs `ingest_versioned` on every document, fills each fact's
`doc_id` from its `document_source`, and writes the dictionary. A fact citing a document that
is not in its batch is an error. Idempotency follows the documents: facts are written only for
documents whose ingest changed, under one extraction job (`kind="adapter:<name>"`) created on
the first batch that has any. Re-running on unchanged input writes nothing: no documents, no
facts, no job.

**The pinned corpus.** Five life-sciences filers, their FY2019–FY2021 10-Ks, 15 filings, in
two tenants. Each filing is pinned by accession number in
`src/fde_kit/rag/adapters/edgar_manifest.json`. Every company yields 18 facts (revenue, net
income and R&D, for each year each 10-K reports).

| Tenant | Companies | Filings |
| --- | --- | --- |
| `pharma-a` | Pfizer (PFE), Gilead (GILD), Vertex (VRTX) | 9 10-Ks; Pfizer's FY2019 one has an Exhibit 13 |
| `pharma-b` | Moderna (MRNA), Regeneron (REGN) | 6 10-Ks |

```python
from pathlib import Path
from fde_kit.rag.adapters import EdgarAdapter, EdgarClient, filings_for, ingest

client = EdgarClient("", cache_dir=Path("evals/fixtures/edgar/cache"), offline=True)
with store.session("pharma-a") as s:
    report = ingest(s, EdgarAdapter(client, filings_for("pharma-a")))
report.documents, report.facts, report.conflict_groups
```

#### What the real data taught us

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
"recent" list had already dropped Pfizer's 2020 10-K, so filings are discovered from the
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

**Offline fixtures.** CI, the tests and `fde rag seed-edgar --offline` never touch sec.gov.
`evals/fixtures/edgar/cache/` holds real responses, trimmed (~1.1 MB): each company's
`companyfacts` cut to the six tags in `FIELDS`, USD, 10-K rows and periods ending 2019–2021
(every row kept exactly as SEC returned it); filing index pages whole; and each filing document
as extracted text truncated to 40,000 characters. The sha256 of the full extracted text is in
the manifest, so the truncation is visible and drift is detectable. Facts come from the XBRL
data, not the text, so truncation does not affect them; it does limit what search can find in
an offline seed to each filing's first 40,000 characters. To regenerate:

```bash
FDE_EDGAR_USER_AGENT="your-name you@example.com" python -m fde_kit.rag.adapters.capture
```

`fde rag edgar-drift` re-fetches every pinned filing live, re-extracts it, and compares the
index and the hashes with the manifest; it exits 1 and names the document if SEC's copy has
changed.

**SEC fair access.** SEC asks automated clients to identify themselves with a contact in the
User-Agent. `EdgarClient` refuses to go online without one (`FDE_EDGAR_USER_AGENT`), stays at
5 requests a second (SEC allows 10), and caches every GET on disk (`FDE_EDGAR_CACHE_DIR`), so a
second run makes no requests. The contact is held in memory only: never logged, never in
`repr`, never written to the cache, the manifest or the fixtures, and a test checks the
fixtures contain no email address. Offline mode needs no contact.

**Local files.** `local_files` loads `.md`, `.txt` and `.htm`/`.html` (through the same
`html_to_text`) from a directory, with the path relative to the root as the source; PDFs are
skipped with a warning. Facts come from an optional `facts.jsonl` at the root, one JSON object
per line naming the document, entity, field, value, unit and period, with `precision` and
`source_ref` honoured when present.

### The attack suite

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

- One partition per tenant per table means seven tables and their indexes per customer.
  Postgres copes with thousands of partitions, and planning stays flat because the store names
  the tenant's own (measured flat from 8 to 208 tenants); tens of thousands of tiny tenants would want a different layout (a database or
  schema per cohort), and pooling small tenants into a shared partition brings the timing
  channel back between them.
- Onboarding a tenant is DDL (`create_tenant`), run by an operator connection, not the app.
- BM25's document frequencies are recomputed from the tenant's chunks on every keyword query.
  Fine at this size; a large tenant would want a maintained term-stats table.

### What's not covered

Per-tenant encryption (see Logs and backups), embedding inversion if vectors leave the database,
catalog metadata (anyone who can run SQL as the app role can count the partitions, read their
row estimates and, with `pg_get_partition_constraintdef`, their tenant ids; that same role can
also just set `app.tenant_id` and read the rows, so the boundary against a SQL-capable attacker
is the role, not the catalog), and the authentication layer: RLS enforces whichever tenant the code sets, so the boundary is
only as good as the line that maps a logged-in user to `tenant_id`. The model provider sees
retrieved passages by design. Facts are loaded by code today (`add_facts`, the demo seed, the source adapters);
`rag_extraction_jobs` is there to record an LLM extraction run, but `fde rag extract` does
not exist yet.

## Baseline results

Golden set (20 cases at the time; the two `citation` cases were added later and have no live
baseline yet) against `claude-sonnet-5-5`, single run, 2026-10-18:

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
offline retrieval eval with its cross-tenant gate, and the offline agent evals (including the
EDGAR set, seeded from the committed fixtures) on every push and PR. On pushes
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
| `FDE_PROMPT` | `default` | System prompt: `prompts/<name>.md`, else the packaged one, or a path to a `.md` file |
| `FDE_PROMPTS_DIR` | `./prompts` | Where named prompts are looked up before the packaged ones |
| `FDE_SYSTEM_PROMPT` | unset | Raw prompt text; overrides `FDE_PROMPT`, recorded as `inline` + hash |
| `FDE_OTEL_ENABLED` | `true` | Export spans |
| `FDE_OTEL_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP/HTTP collector |
| `FDE_OTEL_TENANT_CONTENT` | `false` | Record prompts and tool I/O on spans for tenant-bound runs. Only with a per-tenant access-controlled trace store. |
| `FDE_RUNS_DIR` | `runs` | Where run records are written (`<run_id>.jsonl`). Unredacted local evidence; gitignored. |
| `FDE_RECORD_RUNS` | `false` | Record every `fde run`. `fde rag ask` always records. |
| `FDE_EVAL_PASS_THRESHOLD` | `0.85` | CI gate |
| `FDE_RAG_URL` | derived from admin URL | App connection for retrieval, as `fde_rag_app`. Never a superuser. |
| `FDE_RAG_ADMIN_URL` | `postgresql://fde:fde@localhost:5432/fde` | Used only by `fde rag migrate` / `seed-demo` |
| `FDE_RAG_PARTITION_KEY` | unset | Optional secret; partition names become an HMAC of the tenant id. Fixed at first `migrate`. |
| `FDE_RAG_EMBEDDER` | `bge` | `bge` (bge-small via fastembed) or `hash` (offline, tests) |
| `FDE_EDGAR_USER_AGENT` | unset | Contact SEC requires in the User-Agent, e.g. `"your-name you@example.com"`. Required for live EDGAR; not needed with `--offline`. Never logged or written to disk. |
| `FDE_EDGAR_CACHE_DIR` | `.fde/edgar-cache` | On-disk cache of sec.gov responses; a re-run makes no new requests. |

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
- **Citations are data, not prose.** Tools return `Source`s, the loop numbers them, the answer
  cites numbers, and the check resolves numbers back to `Source`s. There is no step where the
  model's description of its source is trusted.
- **Trace attributes for citations are ids only.** `tool.source_ids` and `tool.citations` go on
  spans even for tenant-bound runs; content stays `[redacted]` there.
- **Replay is deterministic at the tool boundary.** A record stores what the model asked each
  tool and what the tool returned, and a replay serves those rows in order, verifying
  `(name, input)`, so the guarantee is the same evidence path, not the same prose. A request the
  record does not have next is reported as `replay_diverged`, never answered from a different row.
- **A conflict is surfaced, not resolved.** When two sources disagree, every value stays with
  its citation and the agent reports them all; nothing picks a winner. Gilead's
  FY2019 R&D is $9,106M or $4,055M depending on whether acquired in-process R&D counts, and no
  rule in the store can know which one the person asking means.

## What's not here yet

LLM extraction jobs (the table exists; `fde rag extract` does not), pairwise/A-B eval mode, a
customers/ onboarding directory, a one-command cloud deploy, and PDF ingestion in `local_files`
(PDFs are skipped with a warning; there is no PDF dependency).

## License

MIT
