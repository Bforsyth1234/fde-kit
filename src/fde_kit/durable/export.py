"""Export a finished durable run from its Temporal history to a `runs/<run_id>.jsonl` record.

The record format is the one in docs/pr2-record-replay-fork.md section 2, so a durable run and
an in-process `Agent` run produce the same kind of file and replay the same way. Everything
comes from workflow history: the start event carries `AgentRunInput`, each `llm_call` /
`tool_execute` activity carries its typed input and output, the completion event carries
`AgentRunOutput`. Nothing is re-computed and no worker is needed.

`summarize_result` activities are skipped on purpose. The record keeps the tool's raw output
(what the replay needs to re-derive citation numbering); the summarised text the model actually
saw is visible in the next `llm` row's `raw_content`. A tool row whose result was condensed
carries `summarized_from_chars` so a reader knows.

Rows are plain dicts written by a local `_write_jsonl`; the integrator swaps in
`fde_kit.record.RunRecorder` once that module lands.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHandle
from temporalio.converter import DataConverter

from ..citations import cited_numbers
from ..config import get_settings
from .models import (
    AgentRunInput,
    AgentRunOutput,
    LLMCallInput,
    LLMCallOutput,
    ToolCallInput,
    ToolCallOutput,
)

RECORD_VERSION = 1


def run_id_for(workflow_id: str) -> str:
    """`r-` + 12 hex, derived from the workflow id so re-exporting a run overwrites its file."""
    return "r-" + hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write header + rows with the same serializer `RunRecorder` uses, so an exported file and
    a recorded one are byte-compatible line for line."""
    from ..record import _dumps

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")
    return path


def _iso(ts: datetime | None) -> str:
    ts = ts or datetime.now(UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _kind_first(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": kind}
    row.update(fields)
    return row


async def export_run(
    handle: WorkflowHandle[Any, Any],
    *,
    out: Path | None = None,
    runs_dir: Path | None = None,
) -> Path:
    """Walk `handle`'s history and write the run record. Returns the path written.

    `out` names the file exactly; otherwise it is `<runs_dir>/<run_id>.jsonl` with `runs_dir`
    defaulting to `settings.runs_dir` (or `./runs`).
    """
    converter: DataConverter = getattr(handle, "_client", None) and handle._client.data_converter
    if not converter:  # pragma: no cover - handles always carry a client
        from temporalio.converter import default

        converter = default()

    async def decode(payloads: Any, typ: type) -> Any:
        vals = await converter.decode(payloads.payloads, type_hints=[typ])
        return vals[0] if vals else None

    run_input: AgentRunInput | None = None
    run_output: AgentRunOutput | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # scheduled event id -> (activity name, decoded input); completed in history order
    scheduled: dict[int, tuple[str, Any]] = {}
    llm_outputs: list[tuple[LLMCallInput, LLMCallOutput]] = []
    tool_outputs: dict[str, tuple[ToolCallInput, ToolCallOutput]] = {}  # by tool_use_id

    async for ev in handle.fetch_history_events():
        name = EventType.Name(ev.event_type)
        if name == "EVENT_TYPE_WORKFLOW_EXECUTION_STARTED":
            attrs = ev.workflow_execution_started_event_attributes
            run_input = await decode(attrs.input, AgentRunInput)
            started_at = ev.event_time.ToDatetime(tzinfo=UTC)
        elif name == "EVENT_TYPE_ACTIVITY_TASK_SCHEDULED":
            attrs = ev.activity_task_scheduled_event_attributes
            act = attrs.activity_type.name
            if act == "llm_call":
                scheduled[ev.event_id] = (act, await decode(attrs.input, LLMCallInput))
            elif act == "tool_execute":
                scheduled[ev.event_id] = (act, await decode(attrs.input, ToolCallInput))
            # summarize_result and anything else: skipped
        elif name == "EVENT_TYPE_ACTIVITY_TASK_COMPLETED":
            attrs = ev.activity_task_completed_event_attributes
            sched = scheduled.get(attrs.scheduled_event_id)
            if sched is None:
                continue
            act, inp = sched
            if act == "llm_call":
                llm_outputs.append((inp, await decode(attrs.result, LLMCallOutput)))
            elif act == "tool_execute":
                tool_outputs[inp.tool_use_id] = (inp, await decode(attrs.result, ToolCallOutput))
        elif name == "EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED":
            attrs = ev.workflow_execution_completed_event_attributes
            run_output = await decode(attrs.result, AgentRunOutput)
            completed_at = ev.event_time.ToDatetime(tzinfo=UTC)

    if run_input is None:
        raise ValueError(f"workflow {handle.id}: no WorkflowExecutionStarted event in history")
    if run_output is None:
        raise ValueError(f"workflow {handle.id}: run has not completed; nothing to export")

    steps = {s.turn: s for s in run_output.steps}
    model = next((o.model for _, o in llm_outputs if o.model), "")
    tools_seen: list[str] = []
    for _, o in llm_outputs:
        for c in o.tool_calls:
            if c["name"] not in tools_seen:
                tools_seen.append(c["name"])

    run_id = run_id_for(handle.id)
    header = _kind_first(
        "header",
        version=RECORD_VERSION,
        run_id=run_id,
        created_at=_iso(started_at),
        runtime="durable",
        tenant_id=run_input.tenant_id,
        model=model,
        prompt_name=run_output.prompt_name or run_input.prompt_name or "inline",
        prompt_sha256=run_output.prompt_sha256 or run_input.prompt_sha256,
        prompt_text=run_input.system_prompt or "",
        max_turns=run_input.max_turns,
        max_tokens=run_input.max_tokens,
        tools=tools_seen,
        parent_run_id=None,
        fork_at=None,
        workflow_id=handle.id,
    )
    rows: list[dict[str, Any]] = [header]
    rows.append(_kind_first("user", turn=1, text=run_input.prompt, actor="user", by=""))

    for llm_in, llm_out in llm_outputs:
        turn = llm_in.turn
        rows.append(
            _kind_first(
                "llm",
                turn=turn,
                actor="model",
                text=llm_out.text,
                stop_reason=llm_out.stop_reason,
                tool_calls=[
                    {"id": c["id"], "name": c["name"], "input": c["input"]}
                    for c in llm_out.tool_calls
                ],
                raw_content=llm_out.raw_content,
                usage={
                    "input_tokens": llm_out.input_tokens,
                    "output_tokens": llm_out.output_tokens,
                },
            )
        )
        step = steps.get(turn)
        approvals = {a["tool_use_id"]: a for a in (step.approvals if step else [])}
        for i, call in enumerate(llm_out.tool_calls):
            approval = None
            if call["id"] in approvals:
                a = approvals[call["id"]]
                approval = {"outcome": a["outcome"], "by": a["by"], "note": a["note"]}
            step_rec = step.tool_results[i] if step and i < len(step.tool_results) else None
            executed = tool_outputs.get(call["id"])
            if executed is not None:
                tool_in, tool_out = executed
                row = _kind_first(
                    "tool",
                    turn=turn,
                    tool_use_id=call["id"],
                    name=tool_in.name,
                    input=tool_in.input,
                    content=tool_out.content,
                    is_error=tool_out.is_error,
                    sources=tool_out.sources,
                    approval=approval,
                )
            elif step_rec is not None:
                # Gated call that was rejected or timed out: no activity ran, but the model was
                # told so. Keep the row so tool rows line up with `tool_calls` on replay.
                row = _kind_first(
                    "tool",
                    turn=turn,
                    tool_use_id=call["id"],
                    name=call["name"],
                    input=call["input"],
                    content=step_rec["content"],
                    is_error=bool(step_rec.get("is_error", False)),
                    sources=list(step_rec.get("sources") or []),
                    approval=approval,
                )
            else:
                continue
            if step_rec and "summarized_from_chars" in step_rec:
                row["summarized_from_chars"] = step_rec["summarized_from_chars"]
            rows.append(row)

    total_ms = 0
    if started_at and completed_at:
        total_ms = int((completed_at - started_at).total_seconds() * 1000)
    n_sources = len(run_output.sources)
    rows.append(
        _kind_first(
            "result",
            output=run_output.output,
            stopped_reason=run_output.stopped_reason,
            turns=run_output.turns,
            total_ms=total_ms,
            sources=run_output.sources,
            cited=[s["source_id"] for s in run_output.cited],
            fabricated_citations=[
                n for n in cited_numbers(run_output.output) if not (1 <= n <= n_sources)
            ],
        )
    )

    if out is None:
        base = runs_dir or Path(get_settings().runs_dir)
        out = Path(base) / f"{run_id}.jsonl"
    return _write_jsonl(Path(out), rows)
