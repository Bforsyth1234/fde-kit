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
| No double execution | `durable/journal.py` | Tool activities carry an idempotency key `(workflow_id, turn, tool_use_id)`. The result is journaled *before* the activity returns; a retry after a crash-between-complete-and-record is served from the journal. |
| Human approval | `requires_approval=True` on a tool; `approve`/`reject` signals | The run parks on a durable timer. No worker needs to be alive while a person decides. See below. |
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

### Not yet

Mid-run version upgrades (`workflow.patched`) and a Postgres/Redis journal backend. Next two
commits on this branch.

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

`.github/workflows/ci.yml` runs lint, tests and the offline eval on every push and PR. On pushes
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

Retrieval, prompt versioning, pairwise/A-B eval mode, a customers/ onboarding directory, and a
one-command cloud deploy. Those are v1.

## License

MIT
