# fde-kit

**v0.** A slim, opinionated starter kit for putting an LLM agent in front of a customer and
knowing whether it works. Four pieces, nothing else:

| Piece | What it is | Where |
| --- | --- | --- |
| Agent loop | ~150 lines. Prompt → LLM → tools → … → answer. Every step is a span. | `src/fde_kit/agent.py` |
| Tool registry | `@registry.tool` on a typed Python function. Schema derived from the signature. Errors returned to the model, not raised. | `src/fde_kit/tools.py` |
| Tracing | OpenTelemetry → Arize Phoenix (one container, no config). Token counts and tool I/O per span. | `src/fde_kit/tracing.py`, `docker-compose.yml` |
| Eval harness | 20-case golden set, deterministic checks + optional LLM-as-judge, pass-rate gate wired into CI. | `evals/` |

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
