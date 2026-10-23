# PR 2 — Record, replay and fork of a run

Spec and work plan. Branch `feat/record-replay-fork` off `main` (after PR 1 merges) or off
`feat/citations-prompt-versioning`. Wave 1 is contract-first: the record format and the public
API below are frozen here so three workstreams can build against them at the same time.

## 1. What this adds

A finished run becomes a file. The file can be:

- **replayed** — same model, same prompt, same evidence, nothing re-computed. Output is
  byte-identical. Proves the evidence path and citations are reconstructible without the
  database or the model.
- **replayed against the live model** — tools served from the record, LLM called fresh with
  the pinned prompt. The evidence is frozen; the reasoning re-runs; the diff is the experiment.
- **forked** — replay exactly through turn N, then continue live with a new instruction. The
  parent file is never written to. Lineage is recorded.

A durable (Temporal) run exports to the same file from its workflow history, so both runtimes
replay identically.

Determinism, stated plainly: pure replay is exact. Live-LLM replay and forks reproduce the
evidence path exactly and the model's wording approximately. The `cites` checks assert the
former.

## 2. Record format (FROZEN — Wave 1 builds against this)

One file per run: `runs/<run_id>.jsonl`, one JSON object per line, `kind` first. Field order
inside a line is not significant. `run_id` = `r-<12 hex>` (uuid4). Timestamps ISO 8601 UTC.

```jsonc
// line 1, always
{"kind":"header","version":1,"run_id":"r-3f9a1c2e7b4d","created_at":"2026-10-22T23:40:11Z",
 "runtime":"agent",                 // "agent" | "durable"
 "tenant_id":null,                  // or "summit"
 "model":"claude-sonnet-5-5",
 "prompt_name":"default","prompt_sha256":"9abf…","prompt_text":"You are a careful…",
 "max_turns":10,"max_tokens":2048,
 "tools":["search_documents"],      // registry.names() at start
 "parent_run_id":null,"fork_at":null,   // set on a fork: parent file's run_id, turn forked at
 "workflow_id":null}                // durable export only

// the opening user message, and on a fork the injected instruction (turn = N+1)
{"kind":"user","turn":1,"text":"How much is the late fee?","actor":"user","by":""}

// one per LLM call, in order
{"kind":"llm","turn":1,"actor":"model",
 "text":"","stop_reason":"tool_use",
 "tool_calls":[{"id":"toolu_01","name":"search_documents","input":{"query":"late fee"}}],
 "raw_content":[…vendor content blocks, verbatim…],
 "usage":{"input_tokens":812,"output_tokens":41}}

// one per tool execution, in order; content is the tool's RAW output (before the ledger
// rewrites [#i] -> [n]) so replay re-derives numbering deterministically
{"kind":"tool","turn":1,"tool_use_id":"toolu_01","name":"search_documents",
 "input":{"query":"late fee"},"content":"[#1] Lease 101 (…)\n…","is_error":false,
 "sources":[{"source_id":"…","uri":"…","title":"…","doc_id":"…","content_hash":"…","tenant_id":"summit"}],
 "approval":null}                   // durable: {"outcome":"approved","by":"ops","note":""}

// last line, always (even on max_turns)
{"kind":"result","output":"The late fee is $50 [1].","stopped_reason":"end_turn",
 "turns":2,"total_ms":2412,
 "sources":[…Source dicts, citation order…],"cited":["<source_id>", …],
 "fabricated_citations":[]}
```

Rules:

- `tool` rows for one turn follow that turn's `llm` row, in `tool_calls` order.
- A replay matches a requested tool call to the next unconsumed `tool` row by `(name, input)`.
  `tool_use_id` is NOT compared (ids differ across providers and across live reruns).
- A replay of `kind:"llm"` rows is indexed by the number of assistant messages already in the
  conversation, exactly as `MockProvider` does, so a fork's injected user message does not
  shift the index.
- Content in the file is unredacted. The file is local evidence, like a trace with
  `FDE_OTEL_TENANT_CONTENT=1`. The README says so and `runs/` is gitignored.

## 3. Public API (FROZEN)

New module `src/fde_kit/record.py`:

```python
@dataclass
class RunRecord:
    header: dict[str, Any]
    rows: list[dict[str, Any]]          # every non-header line, in file order
    @property
    def run_id(self) -> str
    @property
    def result(self) -> dict[str, Any] | None     # the result row if present
    def llm_rows(self) -> list[dict]; def tool_rows(self) -> list[dict]; def user_rows(self) -> list[dict]
    @classmethod
    def load(cls, path: str | Path) -> "RunRecord"     # validates version and header

class RunRecorder:
    """Streams rows to a file as the run proceeds; header first, result last. Context manager."""
    def __init__(self, path: str | Path, header: dict[str, Any]): ...
    def write(self, row: dict[str, Any]) -> None
    def close(self, result_row: dict[str, Any]) -> None
    def __enter__/__exit__   # __exit__ with an exception still writes a result row with
                             # stopped_reason="error" and the exception type

def new_run_id() -> str
def default_runs_dir(settings) -> Path      # settings.runs_dir or "./runs"
def write_record_from_result(path, header, result: AgentResult) -> Path
    # builds the full file from a finished AgentResult (used by durable export and tests)

class ReplayDiverged(Exception):
    turn: int; expected: dict | None; got: dict      # expected/got = {"name", "input"}

class ReplayProvider:
    """LLMProvider that serves recorded `llm` rows; optionally live after `through_turn`."""
    model: str                                        # header["model"]
    def __init__(self, record: RunRecord, *, through_turn: int | None = None,
                 live: LLMProvider | None = None): ...
    def complete(self, *, system, messages, tools, max_tokens) -> LLMResponse
    # index = count of assistant messages; if through_turn is set and index >= through_turn,
    # delegate to `live` (raise if live is None).

class ReplayRegistry(ToolRegistry):
    """Serves recorded `tool` rows in order, verifying (name, input); live after through_turn."""
    def __init__(self, record: RunRecord, base: ToolRegistry, *,
                 through_turn: int | None = None): ...
    # api_schemas()/names()/get()/requires_approval() delegate to `base` (the model must see the
    # same tool schemas it saw when recorded). execute(): if the current turn <= through_turn
    # (or through_turn is None) serve the next tool row -> ToolResult(content, is_error,
    # sources=tuple(Source.from_dict(...))), raising ReplayDiverged on mismatch; else base.execute.
    # The registry learns the current turn from Agent via `set_turn(turn)` (see hooks).
    tenant_id = base.tenant_id

@dataclass
class ReplaySession:
    """Everything Agent needs to replay or fork. Built by the helpers below, passed to Agent.run."""
    provider: LLMProvider
    registry: ToolRegistry
    through_turn: int | None = None
    inject: tuple[int, str] | None = None     # (turn, text): user message appended BEFORE that turn's LLM call
    parent_run_id: str | None = None

def replay_session(record: RunRecord, base: ToolRegistry) -> ReplaySession
def live_llm_session(record: RunRecord, base: ToolRegistry, live: LLMProvider) -> ReplaySession
def fork_session(record: RunRecord, base: ToolRegistry, live: LLMProvider,
                 at: int, instruction: str) -> ReplaySession
    # replay turns 1..at exactly; inject=(at+1, instruction); live provider and base tools after
```

Changes to `agent.py` (Wave 1-A owns these):

```python
class Agent:
    def __init__(..., recorder: RunRecorder | None = None)   # unchanged otherwise
    def run(self, prompt, *, max_turns=None, session: ReplaySession | None = None,
            record: str | Path | None = None) -> AgentResult
    # session: use session.provider/registry for this run; before turn t, if session.inject and
    #   session.inject[0] == t, append {"role":"user","content": text} and write a user row.
    #   Call registry.set_turn(t) if the registry has it. Catch ReplayDiverged inside _step:
    #   stop the run, stopped_reason="replay_diverged", result.diverged_at = turn,
    #   result.divergence = {"expected":…, "got":…}.
    # record: open a RunRecorder at that path for this run (header built from self + session).
    #   Every llm/tool/user row is written as it happens; result row on exit. Recording and
    #   replaying in one run is allowed (fork writes a new file while reading the parent).

@dataclass
class AgentResult:
    ...
    run_id: str = ""                 # set when recorded
    record_path: str = ""
    diverged_at: int | None = None
    divergence: dict | None = None
```

`AgentResult.stopped_reason` gains the value `"replay_diverged"`. Tracing: the `agent.run`
span gets `agent.run_id`, `agent.replay` (bool), `agent.parent_run_id` when set.

Settings (`config.py`): `runs_dir: str = "runs"` (`FDE_RUNS_DIR`), `record_runs: bool = False`
(`FDE_RECORD_RUNS`; `fde rag ask` ignores it and always records — decision in the design doc).

## 4. CLI (Wave 2-D)

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
fde durable-export WORKFLOW_ID [--out PATH]   # Wave 1-B
```

`--replay` prints the recorded output, then the replayed output, then `identical` / `differs`
(and the first differing line) — for `--live-llm` that line is the whole point. Divergence
prints `replay diverged at turn N: expected <name>(input) got <name>(input)` and exits 3.

## 5. Durable export (Wave 1-B)

`fde_kit/durable/export.py`: `async def export_run(handle, *, out: Path | None) -> Path`.
Walk `handle.fetch_history_events()`:

- `WorkflowExecutionStarted` → `AgentRunInput` (decoded with the client's data converter) →
  header (`runtime:"durable"`, `workflow_id`, prompt fields, tenant) + the `user` row.
- `ActivityTaskScheduled` with activity `llm_call` + its `ActivityTaskCompleted` →
  `LLMCallOutput` → `llm` row. `tool_execute` scheduled+completed → `ToolCallOutput` →
  `tool` row (input from `ToolCallInput`; `sources` straight through; `approval` filled from
  the workflow's recorded approvals via `StepRecord.approvals` in the result).
- `summarize_result` activities are skipped (the record keeps the raw tool content; the
  summarised content is what the model saw, which `llm` rows reflect via the next
  `raw_content`). Note this in the file as `"summarized_from_chars"` on the tool row.
- `WorkflowExecutionCompleted` → `AgentRunOutput` → `result` row.

Replaying a durable export runs through the in-process `Agent` (the point is the file, not the
worker). CLI: `fde durable-export`. Test in `tests/test_durable.py`: run the standard mock
scenario, export, `RunRecord.load`, replay through `Agent` with `replay_session`, assert output
equal and `stopped_reason == "end_turn"`.

## 6. Evals (Wave 2-E)

`evals/golden.jsonl` cases may carry `"replay": "evals/runs/<file>.jsonl"` instead of `"mock"`.
`run_case` then builds `replay_session(RunRecord.load(path), registry)` and runs it; the case's
checks apply to the replayed result. Three cases, category `replay`:

| id | record | checks |
| --- | --- | --- |
| `replay-01` | a recorded `cite-01` mock run | `cites policy:sla`, `cites_only`, `regex ^(?!.*replay_diverged)` via `stopped` check |
| `replay-02` | a two-tool run (`cite-02`) | both `cites`, `cites_only` |
| `replay-03` | a run whose record was edited so the second tool input differs | `stopped: replay_diverged` |

New check type `stopped` (`{"type":"stopped","value":"end_turn"}`) comparing
`result.stopped_reason`. Fixture records are generated by `python -m evals.make_replay_fixtures`
(deterministic: mock provider, fixed run ids, fixed `created_at`) and committed under
`evals/runs/`. `test_evals.py` asserts the fixtures regenerate byte-identical, so a format drift
is caught.

## 7. README (Wave 2-F)

New section **Record, replay, fork** after **Prompt versioning**: the three modes, the CLI
lines, the file format in ten lines, the determinism paragraph verbatim from §1, a `runs/`
retention note (local evidence, unredacted, gitignored, delete on your own schedule), and
`durable-export`. Design-notes bullet: *"Replay is deterministic at the tool boundary."*
Remove record/replay/fork from **What's not here yet**.

## 8. Work plan

Dependencies: every Wave 2 item needs Wave 1-A merged into the branch. 1-B and 1-C need only
this spec (the format and API are frozen). Each agent works in its own git worktree on a
branch `pr2/<letter>`; the integrator merges A first, then rebases B/C/D/E/F on it.

### Wave 1 (parallel, contract-first)

| ID | Workstream | Owns | Done when |
| --- | --- | --- | --- |
| **1-A** | Core record/replay | `record.py`, `agent.py` changes, `config.py` settings, `tests/test_record.py` | `RunRecorder` writes the format in §2; `replay_session` replays a recorded mock run byte-identical; `live_llm_session` serves tools from the record and calls a live (mock) provider; `fork_session` injects the instruction at `at+1` and writes a child file with lineage; `ReplayDiverged` surfaces as `stopped_reason="replay_diverged"` with `diverged_at`; a run recorded from a tenant-bound registry carries `tenant_id` in header and sources; 12+ tests; `ruff` clean |
| **1-B** | Durable export | `durable/export.py`, `fde durable-export`, `tests/test_durable.py::test_export_then_replay` | Export of the standard mock durable run loads with `RunRecord.load`, has the same `llm`/`tool` row count as the in-process run of the same script, and `fde durable-export` writes it. Build against §2 directly: do not import from `record.py` until 1-A lands; write rows as dicts and a local `_write_jsonl`. The integrator swaps in `RunRecorder` at merge |
| **1-C** | `fde runs` commands | `cli.py` (`runs list/tree/show` only), `tests/test_cli_runs.py` | Reads `runs/*.jsonl` with a local loader against §2 (swap to `RunRecord.load` at merge); `list` sorts newest first; `tree` prints parents above and forks below the given run; `show` prints the header and one line per row; works on an empty dir |

### Wave 2 (parallel, after 1-A is on the branch)

| ID | Workstream | Owns | Done when |
| --- | --- | --- | --- |
| **2-D** | `fde run` / `fde rag ask` wiring | `cli.py` (`run` flags, `rag ask` always records), `tests/test_cli_run.py` | All six CLI forms in §4 work with the mock provider; `--replay` prints identical/differs; divergence exits 3; `rag ask` prints the record path; existing `test_rag_isolation.py` untouched |
| **2-E** | Replay evals | `evals/make_replay_fixtures.py`, `evals/runs/*.jsonl`, `run_evals.py` (`replay` key), `scorers.py` (`stopped`), `golden.jsonl` (+3), `test_evals.py` | `fde eval --provider mock --threshold 1.0` passes 25/25; fixtures regenerate byte-identical; `replay-03` fails with `replay_diverged` as designed |
| **2-F** | README + docs | `README.md`, this file's §1–§5 condensed into the README section, `.gitignore` (`runs/`) | README section per §7; case-count references updated (25); `What's not here yet` updated |

### Wave 3 (integrator, sequential)

| ID | Step | Done when |
| --- | --- | --- |
| **3-G** | Merge A, rebase B/C/D/E/F, resolve `cli.py` conflicts (B, C, D all touch it: B adds `durable-export`, C adds the `runs` group, D changes `run`/`rag ask`) | one branch, `ruff check .` clean |
| **3-H** | Verify | `pytest` incl. durable + versioning replay suites; `fde eval --provider mock --threshold 1.0`; round trip: `fde durable-run` (mock worker) → `fde durable-export` → `fde run --replay` identical; `fde rag ask` recording cannot be exercised without pgvector here → CI |
| **3-I** | PR | Push `feat/record-replay-fork`, open PR with the §1 summary, verification list, and the determinism paragraph |

### Conflict map

- `cli.py`: 1-B, 1-C, 2-D. Each adds its own `cmd_*` functions and parser block in a clearly
  delimited region (comment `# --- durable export`, `# --- runs`, `# --- run/replay flags`).
  Keep edits to existing `cmd_run` inside 2-D only.
- `agent.py`, `config.py`, `record.py`: 1-A only.
- `evals/*`: 2-E only. `README.md`: 2-F only.
- `tests/test_durable.py`: 1-B appends one test at the end of the file.

### Out of scope for PR 2

Facts / `query_table` / `fetch_document` (PR 3); adapters and EDGAR (PR 4); replaying inside
Temporal (replay always runs through the in-process `Agent`); a UI.
