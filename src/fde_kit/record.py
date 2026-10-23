"""Record, replay and fork of a run.

A finished run becomes one file: `runs/<run_id>.jsonl`, one JSON object per line, `kind` first.
The header is line 1, the result row is the last line, and between them every `user`, `llm` and
`tool` row in the order it happened. The file can then be

- replayed exactly (`replay_session`): recorded `llm` rows and recorded `tool` rows, nothing
  re-computed, output byte-identical;
- replayed against the live model (`live_llm_session`): tools served from the record, the model
  called fresh with the pinned prompt;
- forked (`fork_session`): replayed exactly through turn N, then a new user instruction is
  injected and the run continues live. The parent file is never written to.

Two small pieces make replay deterministic at the tool boundary:

- `ReplayProvider` indexes `llm` rows by the number of assistant messages already in the
  conversation, exactly as `MockProvider` does, so an injected user message does not shift it.
- `ReplayRegistry` serves `tool` rows in order and verifies `(name, input)`; `tool_use_id` is
  never compared because ids differ across providers and across live reruns. A mismatch raises
  `ReplayDiverged`, which the agent surfaces as `stopped_reason="replay_diverged"`.

Tool rows store the tool's RAW content (before the citation ledger rewrites `[#i]` to `[n]`) so
a replay re-derives the numbering through the same ledger. Content in the file is unredacted:
it is local evidence, like a trace with `FDE_OTEL_TENANT_CONTENT=1`.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .llm import LLMProvider, LLMResponse, ToolCall, Usage
from .tools import Source, ToolRegistry, ToolResult, ToolSpec

if TYPE_CHECKING:  # agent.py imports this module; keep the runtime import one-directional.
    from .agent import AgentResult

RECORD_VERSION = 1


# --------------------------------------------------------------------------- the file


def new_run_id() -> str:
    return f"r-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_runs_dir(settings: Any) -> Path:
    """`settings.runs_dir` or `./runs`."""
    return Path(getattr(settings, "runs_dir", None) or "runs")


def _kind_first(row: dict[str, Any]) -> dict[str, Any]:
    return {"kind": row["kind"], **{k: v for k, v in row.items() if k != "kind"}}


def _dumps(row: dict[str, Any]) -> str:
    return json.dumps(_kind_first(row), ensure_ascii=False, default=str)


@dataclass
class RunRecord:
    header: dict[str, Any]
    rows: list[dict[str, Any]]
    """Every non-header line, in file order."""

    @property
    def run_id(self) -> str:
        return self.header["run_id"]

    @property
    def result(self) -> dict[str, Any] | None:
        for row in reversed(self.rows):
            if row.get("kind") == "result":
                return row
        return None

    def llm_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "llm"]

    def tool_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "tool"]

    def user_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "user"]

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        """Parse a run file. Validates the header line and the format version."""
        path = Path(path)
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"{path}: empty run file")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line 1 is not JSON: {exc}") from exc
        if not isinstance(header, dict) or header.get("kind") != "header":
            raise ValueError(f"{path}: line 1 must be the header row")
        if header.get("version") != RECORD_VERSION:
            raise ValueError(
                f"{path}: unsupported record version {header.get('version')!r} "
                f"(expected {RECORD_VERSION})"
            )
        if not header.get("run_id"):
            raise ValueError(f"{path}: header has no run_id")
        rows = []
        for n, ln in enumerate(lines[1:], start=2):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {n} is not JSON: {exc}") from exc
        return cls(header=header, rows=rows)


class RunRecorder:
    """Streams rows to a file as the run proceeds; header first, result last. Context manager.

    The header is written when the recorder is created. `close(result_row)` writes the last
    line; leaving the `with` block on an exception writes a result row with
    `stopped_reason="error"` and the exception type instead, so the file always ends with one.
    """

    def __init__(self, path: str | Path, header: dict[str, Any]):
        self.path = Path(path)
        self.header = dict(header)
        self.header.setdefault("kind", "header")
        self.header.setdefault("version", RECORD_VERSION)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = datetime.now(UTC)
        self._llm_rows = 0
        self._closed = False
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(_dumps(self.header) + "\n")
        self._fh.flush()

    @property
    def run_id(self) -> str:
        return self.header["run_id"]

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, row: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("RunRecorder is closed")
        if row.get("kind") == "llm":
            self._llm_rows += 1
        self._fh.write(_dumps(row) + "\n")
        self._fh.flush()

    def close(self, result_row: dict[str, Any]) -> None:
        if self._closed:
            return
        row = dict(result_row)
        row.setdefault("kind", "result")
        self._fh.write(_dumps(row) + "\n")
        self._fh.close()
        self._closed = True

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._closed:
            return
        if exc_type is None:
            self.close(_result_row_dict("", "end_turn", self._llm_rows, self._elapsed_ms()))
            return
        self.close(
            {
                **_result_row_dict("", "error", self._llm_rows, self._elapsed_ms()),
                "error": exc_type.__name__,
                "error_message": str(exc),
            }
        )

    def _elapsed_ms(self) -> int:
        return int((datetime.now(UTC) - self._t0).total_seconds() * 1000)


def _result_row_dict(
    output: str,
    stopped_reason: str,
    turns: int,
    total_ms: float,
    sources: list[dict[str, Any]] | None = None,
    cited: list[str] | None = None,
    fabricated: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "result",
        "output": output,
        "stopped_reason": stopped_reason,
        "turns": turns,
        "total_ms": int(total_ms),
        "sources": sources or [],
        "cited": cited or [],
        "fabricated_citations": fabricated or [],
    }


def result_row(result: AgentResult) -> dict[str, Any]:
    """The `result` row for a finished `AgentResult`."""
    row = _result_row_dict(
        result.output,
        result.stopped_reason,
        result.turns,
        result.total_ms,
        [s.to_dict() for s in result.sources],
        [s.source_id for s in result.cited],
        list(result.fabricated_citations),
    )
    if result.diverged_at is not None:
        row["diverged_at"] = result.diverged_at
        row["divergence"] = result.divergence
    return row


def user_row(turn: int, text: str, by: str = "") -> dict[str, Any]:
    return {"kind": "user", "turn": turn, "text": text, "actor": "user", "by": by}


def llm_row(turn: int, resp: LLMResponse) -> dict[str, Any]:
    return {
        "kind": "llm",
        "turn": turn,
        "actor": "model",
        "text": resp.text,
        "stop_reason": resp.stop_reason,
        "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls],
        "raw_content": copy.deepcopy(resp.raw_content),
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def tool_row(turn: int, call: ToolCall, res: ToolResult) -> dict[str, Any]:
    """`res.content` must be the tool's raw output, before the citation ledger rewrite."""
    return {
        "kind": "tool",
        "turn": turn,
        "tool_use_id": call.id,
        "name": call.name,
        "input": call.input,
        "content": res.content,
        "is_error": res.is_error,
        "sources": [s.to_dict() for s in res.sources],
        "approval": None,
    }


def write_record_from_result(path: str | Path, header: dict[str, Any], result: AgentResult) -> Path:
    """Build the full file from a finished `AgentResult` (durable export, tests).

    Tool rows take `raw_content` from each step's tool result when the loop recorded it, and
    fall back to the ledger-rewritten content otherwise.
    """
    path = Path(path)
    with RunRecorder(path, header) as rec:
        rec.write(user_row(1, result.prompt))
        for step in result.steps:
            raw = step.raw_content or _reconstruct_raw(step.assistant_text, step.tool_calls)
            rec.write(
                {
                    "kind": "llm",
                    "turn": step.turn,
                    "actor": "model",
                    "text": step.assistant_text,
                    "stop_reason": step.stop_reason
                    or ("tool_use" if step.tool_calls else "end_turn"),
                    "tool_calls": [
                        {"id": c.get("id", ""), "name": c["name"], "input": c["input"]}
                        for c in step.tool_calls
                    ],
                    "raw_content": raw,
                    "usage": {
                        "input_tokens": step.usage.input_tokens,
                        "output_tokens": step.usage.output_tokens,
                    },
                }
            )
            for call, tr in zip(step.tool_calls, step.tool_results, strict=False):
                rec.write(
                    {
                        "kind": "tool",
                        "turn": step.turn,
                        "tool_use_id": call.get("id", ""),
                        "name": call["name"],
                        "input": call["input"],
                        "content": tr.get("raw_content", tr["content"]),
                        "is_error": bool(tr.get("is_error")),
                        "sources": list(tr.get("sources", [])),
                        "approval": tr.get("approval"),
                    }
                )
        rec.close(result_row(result))
    return path


def _reconstruct_raw(text: str, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    if text:
        raw.append({"type": "text", "text": text})
    for c in tool_calls:
        raw.append(
            {"type": "tool_use", "id": c.get("id", ""), "name": c["name"], "input": c["input"]}
        )
    return raw


# --------------------------------------------------------------------------- replay


class ReplayDiverged(Exception):
    """The run asked for something the record does not have next.

    `expected` is the next unconsumed record entry as `{"name", "input"}` (None when the record
    is exhausted); `got` is what the run asked for in the same shape.
    """

    def __init__(self, turn: int, expected: dict[str, Any] | None, got: dict[str, Any]):
        self.turn = turn
        self.expected = expected
        self.got = got
        super().__init__(self._describe())

    def _describe(self) -> str:
        exp = (
            f"{self.expected['name']}({json.dumps(self.expected['input'], sort_keys=True)})"
            if self.expected
            else "<end of record>"
        )
        got = f"{self.got['name']}({json.dumps(self.got['input'], sort_keys=True)})"
        return f"replay diverged at turn {self.turn}: expected {exp} got {got}"


class ReplayProvider:
    """LLMProvider that serves recorded `llm` rows; optionally live after `through_turn`.

    The row index is the count of assistant messages already in `messages`. If `through_turn`
    is set and the index reaches it, the call is delegated to `live` instead.
    """

    def __init__(
        self,
        record: RunRecord,
        *,
        through_turn: int | None = None,
        live: LLMProvider | None = None,
    ):
        self.record = record
        self.model: str = record.header.get("model", "")
        self.through_turn = through_turn
        self.live = live
        self._rows = record.llm_rows()

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        idx = sum(1 for m in messages if m.get("role") == "assistant")
        if self.through_turn is not None and idx >= self.through_turn:
            if self.live is None:
                raise RuntimeError(
                    f"ReplayProvider: turn {idx + 1} is past through_turn={self.through_turn} "
                    "and no live provider was given"
                )
            return self.live.complete(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        if idx >= len(self._rows):
            raise ReplayDiverged(
                idx + 1, None, {"name": "<llm call>", "input": {"assistant_messages": idx}}
            )
        return _response_from_row(self._rows[idx])


def _response_from_row(row: dict[str, Any]) -> LLMResponse:
    usage = row.get("usage") or {}
    return LLMResponse(
        text=row.get("text", ""),
        tool_calls=[
            ToolCall(id=tc.get("id", ""), name=tc["name"], input=copy.deepcopy(tc["input"]))
            for tc in row.get("tool_calls", [])
        ],
        stop_reason=row.get("stop_reason", "end_turn"),
        usage=Usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
        raw_content=copy.deepcopy(row.get("raw_content", [])),
    )


class ReplayRegistry(ToolRegistry):
    """Serves recorded `tool` rows in order, verifying `(name, input)`; live after `through_turn`.

    Schemas, names, lookups and approval flags come from `base`: the model must see the same
    tool schemas it saw when recorded. The agent tells the registry which turn it is on via
    `set_turn(turn)`; while `turn <= through_turn` (or `through_turn` is None) the next
    unconsumed tool row is served, otherwise `base.execute` runs the real tool.
    """

    def __init__(
        self,
        record: RunRecord,
        base: ToolRegistry,
        *,
        through_turn: int | None = None,
    ):
        self.record = record
        self.base = base
        self.through_turn = through_turn
        self._tools = base._tools
        self.tenant_id = base.tenant_id
        self._rows = record.tool_rows()
        self._next = 0
        self._turn = 1

    def set_turn(self, turn: int) -> None:
        self._turn = turn

    @property
    def consumed(self) -> int:
        """How many recorded tool rows have been served so far."""
        return self._next

    def names(self) -> list[str]:
        return self.base.names()

    def get(self, name: str) -> ToolSpec:
        return self.base.get(name)

    def api_schemas(self) -> list[dict[str, Any]]:
        return self.base.api_schemas()

    def requires_approval(self, name: str) -> bool:
        return self.base.requires_approval(name)

    def execute(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        if self.through_turn is not None and self._turn > self.through_turn:
            return self.base.execute(name, raw_input)
        got = {"name": name, "input": raw_input}
        if self._next >= len(self._rows):
            raise ReplayDiverged(self._turn, None, got)
        row = self._rows[self._next]
        expected = {"name": row["name"], "input": row["input"]}
        if row["name"] != name or row["input"] != raw_input:
            raise ReplayDiverged(self._turn, expected, got)
        self._next += 1
        return ToolResult(
            row["content"],
            is_error=bool(row.get("is_error")),
            sources=tuple(Source.from_dict(d) for d in row.get("sources", [])),
        )


@dataclass
class ReplaySession:
    """Everything Agent needs to replay or fork. Built by the helpers below, passed to Agent.run."""

    provider: LLMProvider
    registry: ToolRegistry
    through_turn: int | None = None
    inject: tuple[int, str] | None = None
    """(turn, text): user message appended BEFORE that turn's LLM call."""
    parent_run_id: str | None = None

    @property
    def fork_at(self) -> int | None:
        return self.inject[0] - 1 if self.inject else None


def recorded_prompt(record: RunRecord) -> str:
    """The opening user prompt of a record (its first `user` row)."""
    rows = record.user_rows()
    return rows[0].get("text", "") if rows else ""


def replay_session(record: RunRecord, base: ToolRegistry) -> ReplaySession:
    """Exact replay: every LLM and tool row from the record, nothing re-computed."""
    return ReplaySession(
        provider=ReplayProvider(record),
        registry=ReplayRegistry(record, base),
        parent_run_id=record.run_id,
    )


def live_llm_session(record: RunRecord, base: ToolRegistry, live: LLMProvider) -> ReplaySession:
    """Tools served from the record, every LLM call made against `live`."""
    return ReplaySession(
        provider=ReplayProvider(record, through_turn=0, live=live),
        registry=ReplayRegistry(record, base),
        through_turn=0,
        parent_run_id=record.run_id,
    )


def fork_session(
    record: RunRecord,
    base: ToolRegistry,
    live: LLMProvider,
    at: int,
    instruction: str,
) -> ReplaySession:
    """Replay turns 1..`at` exactly, inject `instruction` as a user message before turn
    `at + 1`, then continue with `live` and the real tools in `base`."""
    if at < 0:
        raise ValueError("fork_session: `at` must be >= 0")
    return ReplaySession(
        provider=ReplayProvider(record, through_turn=at, live=live),
        registry=ReplayRegistry(record, base, through_turn=at),
        through_turn=at,
        inject=(at + 1, instruction),
        parent_run_id=record.run_id,
    )
