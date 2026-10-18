"""The agent loop as a Temporal workflow.

This is `Agent.run` from agent.py with the two side effects moved into activities. Compare them
side by side: the control flow is identical. What changes:

- The loop's state (messages, steps, usage) is workflow state, which Temporal persists as event
  history. Kill the worker at any point and a new worker replays history to rebuild it exactly.
- Each LLM call and each tool call is an activity with timeouts, heartbeats and a retry policy.
- Tool calls carry an idempotency key derived from (workflow id, turn, tool_use id). Same inputs
  on retry -> same key -> journal hit -> no double execution.
- A `status` query lets anyone ask a running agent where it is without touching the worker.

Determinism rules apply inside this file: no I/O, no clocks, no randomness, no network. All of
that lives in activities.py.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import AgentActivities
    from .models import (
        AgentRunInput,
        AgentRunOutput,
        LLMCallInput,
        RunStatus,
        StepRecord,
        ToolCallInput,
    )

DEFAULT_SYSTEM = (
    "You are a careful assistant. Use the provided tools when they help answer "
    "precisely. When you have the answer, reply with it directly and concisely."
)

LLM_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)
TOOL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=10),
    maximum_attempts=5,
)


@workflow.defn(name="AgentWorkflow")
class AgentWorkflow:
    def __init__(self) -> None:
        self._turn = 0
        self._phase = "llm"
        self._tools_called: list[str] = []
        self._current_tool: str | None = None

    @workflow.query
    def status(self) -> RunStatus:
        return RunStatus(
            turn=self._turn,
            phase=self._phase,
            tools_called=list(self._tools_called),
            current_tool=self._current_tool,
        )

    @workflow.run
    async def run(self, inp: AgentRunInput) -> AgentRunOutput:
        system = inp.system_prompt or DEFAULT_SYSTEM
        messages: list[dict] = [{"role": "user", "content": inp.prompt}]
        steps: list[StepRecord] = []
        in_tok = out_tok = 0
        wf_id = workflow.info().workflow_id

        for turn in range(1, inp.max_turns + 1):
            self._turn, self._phase = turn, "llm"

            llm = await workflow.execute_activity(
                AgentActivities.llm_call,
                LLMCallInput(
                    system=system, messages=messages, max_tokens=inp.max_tokens, turn=turn
                ),
                start_to_close_timeout=timedelta(minutes=3),
                heartbeat_timeout=timedelta(seconds=20),
                retry_policy=LLM_RETRY,
            )
            in_tok += llm.input_tokens
            out_tok += llm.output_tokens
            messages.append({"role": "assistant", "content": llm.raw_content})

            if not llm.tool_calls:
                steps.append(StepRecord(turn=turn, assistant_text=llm.text))
                self._phase = "done"
                return AgentRunOutput(
                    output=llm.text,
                    steps=steps,
                    stopped_reason="end_turn",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )

            self._phase = "tools"
            result_blocks: list[dict] = []
            recorded: list[dict] = []
            for call in llm.tool_calls:
                self._current_tool = call["name"]
                self._tools_called.append(call["name"])
                res = await workflow.execute_activity(
                    AgentActivities.tool_execute,
                    ToolCallInput(
                        name=call["name"],
                        input=call["input"],
                        idempotency_key=f"{wf_id}:{turn}:{call['id']}",
                        tool_use_id=call["id"],
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    heartbeat_timeout=timedelta(seconds=10),
                    retry_policy=TOOL_RETRY,
                )
                block = {"type": "tool_result", "tool_use_id": call["id"], "content": res.content}
                if res.is_error:
                    block["is_error"] = True
                result_blocks.append(block)
                recorded.append(
                    {
                        "name": call["name"],
                        "content": res.content,
                        "is_error": res.is_error,
                        "from_journal": res.from_journal,
                    }
                )
            self._current_tool = None
            messages.append({"role": "user", "content": result_blocks})
            steps.append(
                StepRecord(
                    turn=turn,
                    assistant_text=llm.text,
                    tool_calls=llm.tool_calls,
                    tool_results=recorded,
                )
            )

        self._phase = "done"
        return AgentRunOutput(
            output=steps[-1].assistant_text if steps else "",
            steps=steps,
            stopped_reason="max_turns",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
