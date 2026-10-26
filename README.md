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

What building and testing it turned up (live eval scores, real SEC data, the isolation
attacks, a measured timing leak) is in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

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
fde runs show RUN_ID [--full]             # header + one line per row; --full adds tool inputs (SQL) and outputs
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
`cites_only`, `cites_any`, `cites_one_of`, `stopped`, `amount`, `no_amount`, `date`,
`conflict_reported`, `judge`. A case passes only if every check passes. `cites <id>` passes when
the answer cites, by `[n]`, a Source whose `source_id`, `doc_id` or `uri` is `<id>`; naming the
document in prose does not count. `cites_one_of [ids]` passes when it cites any of them (for a
clause every lease repeats). `cites_only` passes when no citation in the answer is fabricated.
`stopped <reason>` compares `result.stopped_reason` (`end_turn`, `max_turns`, `replay_diverged`).

Numbers and dates are checked by value, not by string. `amount "$16,071.7 million"` parses every
amount in the answer, scale words included, so "$16.07 billion" and "USD 16,071,700,000" match
too (relative tolerance 0.25%, tight enough that Pfizer's 41,172 and 40,905 stay distinct).
`no_amount` is the leak check: the other tenant's number must not appear in any format, with a
wider 1% tolerance so it errs toward flagging. `date "2026-12-01"` also accepts "December 1,
2026". `conflict_reported [values]` passes when every value is present (value-aware) and the
answer says the sources disagree (conflict, disagree, differ, restate, revise, inconsistent,
discrepancy, in any form). Wording that a string check cannot judge fairly, such as whether a
refusal says the company is not on file or whether a precision difference is explained as
rounding, goes to a `judge` rubric; judges run only against a live model, so mock runs check
everything else. The judge sees the tool outputs the model saw (capped at 12,000
characters), so it can tell a claim quoted from a document from one the model brought in itself.

The report shows pass rate overall and per category, p50/p95 latency, token
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
documents), and two ask one tenant for another tenant's unit (`cites_only`, `no_amount` for the
other tenant's figure, and a judge rubric that the answer says not found). CI seeds the demo and runs it in mock mode with a pass threshold of 1.0.

`fde eval --set edgar` runs a third set, `evals/golden_edgar.jsonl`: 12 cases over the pinned
EDGAR corpus (see Source adapters and SEC EDGAR), seeded offline from the committed fixtures.
Three `table-fact` cases reach one number through `query_table` (Moderna FY2021 revenue,
Gilead FY2021 net income, Regeneron FY2021 revenue) and cite the filing. Three
`table-conflict` cases hit a real restatement (Pfizer FY2019 revenue, Gilead FY2019 R&D,
Regeneron FY2019 revenue): `conflict_reported` every value, `cites` every filing. Two
`precision` cases (Vertex FY2019 revenue, Moderna FY2020 net loss) assert the opposite: the
value is stated, and a judge rubric checks the answer treats the gap as rounding, not a conflict. Two
`doc-lookup` cases answer from the first 40,000 characters of a filing, and two `cross-tenant`
cases ask `pharma-b` for Pfizer and `pharma-a` for Moderna and expect not found, with the
other tenant's figure absent in any format. CI runs it in
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

A crash can land anywhere in a step: during the LLM call, mid-tool, after a tool finished but
before Temporal recorded it, or between steps. The idempotency journal covers every case but
one: a crash *during* a non-idempotent tool, before its result is journaled, re-runs the tool.
The row-by-row table is in [Findings](docs/FINDINGS.md#durable-runs-what-a-crash-does-at-each-point).

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
make rag-eval                     # retrieval numbers (docs/FINDINGS.md)
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
| `query_table(sql: str)` | a header line, then one row per line. A row whose `fact_id` is found in the database gets `[#i]`, a Source, and ` (source: <document title>)`; a fact in a conflict group gets ` conflict=true`, whether or not the SQL selected `conflict_group_id`; `SELECT count(*)` rows get neither. More than 200 rows: `... (truncated at 200 rows; add a WHERE or LIMIT)` | Looked up from the database by `fact_id` (`TenantSession.facts_by_id`, same tenant session): `source_id` = `fact_id`, `uri` = the document's URL, `doc_id`, `content_hash` = the document's hash, title `<entity> <field> <period>`, `span` = `fact:{fact_id}`. The model's SELECT does not decide provenance. A `fact_id` the database does not know (a made-up literal, or another tenant's id hidden by RLS) gets no `[#i]` and no Source, and its line ends with ` (unverified: fact_id not found)`. |

The `query_table` description is built once per registry: the docstring, then a paragraph
naming `rag_facts`, `rag_entities` and `rag_documents` with their columns, what the guard does
not allow (`information_schema`, `pg_*`), and a worked example query (a test checks the guard
accepts it), then one `- field (unit): definition` line per `rag_dictionary` row. One query per
request, not per call. The example and the inline `(source: …)` labels came from a live run in
which the model spent four `query_table` calls on one question, one of them only to learn which
filing each number came from, and still cited facts with no document attached.

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
facts' precision since PR 4; see
[precision](docs/FINDINGS.md#what-real-sec-data-taught-us)), all of them and
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

Pfizer's FY2019 revenue is restated twice ($51,750M → $41,172M → $40,905M), Gilead's R&D
changes XBRL tag between filings, later filings round earlier years more coarsely, and Pfizer's
FY2019 statements live only in Exhibit 13. Each of those shaped the adapter (conflict groups,
precision-aware agreement, per-filing tag priority, EX-13 ingest, `source_ref`). Details and
numbers: [What real SEC data taught us](docs/FINDINGS.md#what-real-sec-data-taught-us).

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

`tests/test_rag_isolation.py`, `tests/test_rag_facts.py`, `tests/test_tracing_tenants.py`,
`tests/test_sqlguard.py` and `tests/test_rag_tool.py`: 100+ tests against a real Postgres, on a
corpus where three property managers share one lease template, so any leak reads like a correct
answer with the wrong number. They cover the data, catalogs, ids, timing, logs, traces,
`query_table` SQL, and misconfiguration (the store refuses to start). Demonstration tests
reproduce real failures under configurations the store refuses to use.

Measured results:
- [The attack table, demonstrations and mutation check](docs/FINDINGS.md#tenant-isolation-the-attack-suite)
- [The timing leak](docs/FINDINGS.md#the-timing-leak-measured): on a shared index, query work grew 136× with another tenant's data; partitioned, it doesn't move.
- [Logs and backups](docs/FINDINGS.md#logs-and-backups): parameter logging is pinned off; why "deleted" doesn't reach backups.
- [Two things I didn't expect](docs/FINDINGS.md#two-things-i-didnt-expect): the planner can't see the tenant, and fusion only works when scores mean something.
- [Retrieval numbers](docs/FINDINGS.md#retrieval-numbers): hybrid hit@1 0.96, cross-tenant hits 0.
- [Costs and limits](docs/FINDINGS.md#costs-and-limits).

## Baseline results

Live, against `claude-sonnet-5-5`, October 2026:

| Set | Pass rate |
| --- | --- |
| golden | **24 / 25** |
| rag (leases) | **12 / 12** |
| edgar (SEC filings) | **12 / 12** |

No fabricated citations and no cross-tenant figures in any set. Per-category numbers, latency,
tokens, the one known failure, and why the first live run scored 8/12 and 5/12:
[Live eval results](docs/FINDINGS.md#live-eval-results).

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
