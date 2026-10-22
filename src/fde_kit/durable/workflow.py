"""The agent loop as a Temporal workflow.

This is `Agent.run` from agent.py with the two side effects moved into activities. Compare them
side by side: the control flow is identical. What changes:

- The loop's state (messages, steps, usage) is workflow state, which Temporal persists as event
  history. Kill the worker at any point and a new worker replays history to rebuild it exactly.
- Each LLM call and each tool call is an activity with timeouts, heartbeats and a retry policy.
- Tool calls carry an idempotency key derived from (workflow id, turn, tool_use id). Same inputs
  on retry -> same key -> journal hit -> no double execution.
- Tools marked `requires_approval` pause the run until an `approve` or `reject` signal arrives.
  The wait is a Temporal timer, not a process: no worker needs to be alive while a human
  thinks about it, and the run survives any number of restarts in the meantime.
- A `status` query lets anyone ask a running agent where it is without touching the worker.
- Code changes that alter the sequence of commands are guarded by `workflow.patched(...)`.
  A run that started on the old code is pinned to the old branch for its whole lifetime, even
  the steps executed after the upgrade; runs started on the new code take the new branch from
  step one. Both kinds complete on the same worker without a non-determinism error. See
  VERSIONING below.

VERSIONING
  v1  initial: llm_call / tool_execute / approval gates
  v2  "v2-summarize-results": long tool results are condensed by a summarize_result activity
      before re-entering the model's context. Adds one activity per long result, which is a
      new command in the sequence, hence the patch.

  Rollout: ship v2 with `patched` (now). Once no v1 runs are open, switch to
  `workflow.deprecate_patch("v2-summarize-results")` and drop the else-branch. After v1
  histories age out of retention, remove the deprecate call. Three deploys, zero failed runs.

Determinism rules apply inside this file: no I/O, no clocks, no randomness, no network. All of
that lives in activities.py.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..citations import CitationLedger
    from ..prompts import prompt_hash
    from ..tools import Source
    from .activities import AgentActivities
    from .models import (
        AgentRunInput,
        AgentRunOutput,
        ApprovalDecision,
        LLMCallInput,
        PendingApproval,
        RunStatus,
        StepRecord,
        SummarizeInput,
        ToolCallInput,
    )

CODE_VERSION = 2
PATCH_V2 = "v2-summarize-results"
SUMMARIZE_OVER_CHARS = 1500
SUMMARIZE_TO_CHARS = 600

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
        self._pending: PendingApproval | None = None
        self._decisions: dict[str, ApprovalDecision] = {}

    # ------------------------------------------------------------------ queries

    @workflow.query
    def status(self) -> RunStatus:
        return RunStatus(
            turn=self._turn,
            phase=self._phase,
            tools_called=list(self._tools_called),
            current_tool=self._current_tool,
            pending_approval=self._pending,
        )

    @workflow.query
    def pending_approval(self) -> PendingApproval | None:
        return self._pending

    # ------------------------------------------------------------------ signals

    @workflow.signal
    def approve(self, tool_use_id: str, by: str = "", note: str = "") -> None:
        self._decisions[tool_use_id] = ApprovalDecision(tool_use_id, True, by, note)

    @workflow.signal
    def reject(self, tool_use_id: str, by: str = "", note: str = "") -> None:
        self._decisions[tool_use_id] = ApprovalDecision(tool_use_id, False, by, note)

    # ------------------------------------------------------------------ run

    @workflow.run
    async def run(self, inp: AgentRunInput) -> AgentRunOutput:
        system = inp.system_prompt or DEFAULT_SYSTEM
        # sha256 of a string is deterministic, so this is safe inside the workflow and lets a
        # history that predates prompt versioning still report what prompt it ran.
        prompt_name = inp.prompt_name or "inline"
        prompt_sha = inp.prompt_sha256 or prompt_hash(system)
        messages: list[dict] = [{"role": "user", "content": inp.prompt}]
        steps: list[StepRecord] = []
        in_tok = out_tok = 0
        wf_id = workflow.info().workflow_id
        extra_gated = set(inp.require_approval_for)
        ledger = CitationLedger()

        for turn in range(1, inp.max_turns + 1):
            self._turn, self._phase = turn, "llm"

            llm = await workflow.execute_activity(
                AgentActivities.llm_call,
                LLMCallInput(
                    system=system,
                    messages=messages,
                    max_tokens=inp.max_tokens,
                    turn=turn,
                    tenant_id=inp.tenant_id,
                ),
                start_to_close_timeout=timedelta(minutes=3),
                heartbeat_timeout=timedelta(seconds=20),
                retry_policy=LLM_RETRY,
            )
            in_tok += llm.input_tokens
            out_tok += llm.output_tokens
            messages.append({"role": "assistant", "content": llm.raw_content})

            if not llm.tool_calls:
                steps.append(
                    StepRecord(turn=turn, assistant_text=llm.text, code_version=CODE_VERSION)
                )
                self._phase = "done"
                return AgentRunOutput(
                    output=llm.text,
                    steps=steps,
                    stopped_reason="end_turn",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    code_version=CODE_VERSION,
                    sources=[s.to_dict() for s in ledger.sources],
                    prompt_name=prompt_name,
                    prompt_sha256=prompt_sha,
                )

            self._phase = "tools"
            result_blocks: list[dict] = []
            recorded: list[dict] = []
            approvals: list[dict] = []
            step_version = 1
            for call in llm.tool_calls:
                self._current_tool = call["name"]
                self._tools_called.append(call["name"])

                gated = call.get("requires_approval", False) or call["name"] in extra_gated
                if gated:
                    decision = await self._await_approval(call, turn, inp.approval_timeout_seconds)
                    approvals.append(
                        {
                            "tool_use_id": call["id"],
                            "name": call["name"],
                            "outcome": "approved"
                            if decision.approved
                            else ("timeout" if decision.by == "timeout" else "rejected"),
                            "by": decision.by,
                            "note": decision.note,
                        }
                    )
                    if not decision.approved:
                        reason = decision.note or (
                            "approval timed out" if decision.by == "timeout" else "no reason given"
                        )
                        content = (
                            f"Tool call '{call['name']}' was not approved by a human reviewer "
                            f"({reason}). Do not retry it; explain to the user or find another way."
                        )
                        result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": content,
                                "is_error": True,
                            }
                        )
                        recorded.append(
                            {
                                "name": call["name"],
                                "content": content,
                                "is_error": True,
                                "from_journal": False,
                            }
                        )
                        continue

                res = await workflow.execute_activity(
                    AgentActivities.tool_execute,
                    ToolCallInput(
                        name=call["name"],
                        input=call["input"],
                        idempotency_key=f"{wf_id}:{turn}:{call['id']}",
                        tool_use_id=call["id"],
                        tenant_id=inp.tenant_id,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    heartbeat_timeout=timedelta(seconds=10),
                    retry_policy=TOOL_RETRY,
                )
                # Pure string rewrite of local `[#i]` markers to run-wide `[n]`; deterministic,
                # and a no-op on histories whose tool outputs carry no sources.
                content, numbers = ledger.add(
                    res.content, [Source.from_dict(d) for d in res.sources]
                )
                summarized_from: int | None = None
                # v2: condense long results. `patched` is sticky per run: a run that first
                # evaluates it while replaying a history with no marker is pinned to the
                # else-branch for good, so a deploy never changes an in-flight conversation's
                # shape; a run that first evaluates it live records a marker and stays on v2.
                if workflow.patched(PATCH_V2):
                    step_version = 2
                    if not res.is_error and len(content) > SUMMARIZE_OVER_CHARS:
                        summ = await workflow.execute_activity(
                            AgentActivities.summarize_result,
                            SummarizeInput(
                                tool_name=call["name"],
                                content=content,
                                max_chars=SUMMARIZE_TO_CHARS,
                                tenant_id=inp.tenant_id,
                            ),
                            start_to_close_timeout=timedelta(minutes=2),
                            heartbeat_timeout=timedelta(seconds=20),
                            retry_policy=LLM_RETRY,
                        )
                        if summ.summarized:
                            content = summ.content
                            summarized_from = summ.original_chars
                else:
                    step_version = 1

                block = {"type": "tool_result", "tool_use_id": call["id"], "content": content}
                if res.is_error:
                    block["is_error"] = True
                result_blocks.append(block)
                rec = {
                    "name": call["name"],
                    "content": content,
                    "is_error": res.is_error,
                    "from_journal": res.from_journal,
                    "sources": res.sources,
                    "citations": numbers,
                }
                if summarized_from is not None:
                    rec["summarized_from_chars"] = summarized_from
                recorded.append(rec)
            self._current_tool = None
            messages.append({"role": "user", "content": result_blocks})
            steps.append(
                StepRecord(
                    turn=turn,
                    assistant_text=llm.text,
                    tool_calls=llm.tool_calls,
                    tool_results=recorded,
                    approvals=approvals,
                    code_version=step_version,
                )
            )

        self._phase = "done"
        return AgentRunOutput(
            output=steps[-1].assistant_text if steps else "",
            steps=steps,
            stopped_reason="max_turns",
            input_tokens=in_tok,
            output_tokens=out_tok,
            code_version=CODE_VERSION,
            sources=[s.to_dict() for s in ledger.sources],
            prompt_name=prompt_name,
            prompt_sha256=prompt_sha,
        )

    # ------------------------------------------------------------------ helpers

    async def _await_approval(self, call: dict, turn: int, timeout_s: int) -> ApprovalDecision:
        """Park the run until a human decides, or the timeout elapses.

        `wait_condition` is a durable wait: Temporal persists a timer and the workflow task
        simply ends. Nothing runs, nothing polls, no worker needs to exist until a signal or the
        timer wakes it. A decision that arrived *before* we started waiting (fast reviewer, or a
        replay after restart) is honoured immediately because the condition checks the dict.
        """
        tool_use_id = call["id"]
        self._pending = PendingApproval(
            tool_use_id=tool_use_id, name=call["name"], input=call["input"], turn=turn
        )
        self._phase = "awaiting_approval"
        try:
            await workflow.wait_condition(
                lambda: tool_use_id in self._decisions, timeout=timedelta(seconds=timeout_s)
            )
            decision = self._decisions[tool_use_id]
        except TimeoutError:
            decision = ApprovalDecision(tool_use_id, False, by="timeout", note="approval timed out")
        finally:
            self._pending = None
            self._phase = "tools"
        return decision
