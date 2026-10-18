"""Activities: the two side-effecting steps of the agent loop.

Activities are where non-determinism is allowed. The LLM call and the tool execution both live
here; the workflow only decides what to call next. Both activities heartbeat while they run, so
if the worker process dies mid-step Temporal notices within `heartbeat_timeout` and reschedules
the activity on another worker instead of waiting for the full `start_to_close_timeout`.

Tool execution consults the idempotency journal first. See journal.py for why.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from temporalio import activity

from ..llm import LLMProvider
from ..tools import ToolRegistry
from .journal import ToolJournal
from .models import LLMCallInput, LLMCallOutput, ToolCallInput, ToolCallOutput

T = TypeVar("T")


class AgentActivities:
    """Holds the live provider, registry and journal. One instance per worker process."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        journal: ToolJournal | None = None,
        heartbeat_every: float = 1.0,
    ):
        self.provider = provider
        self.registry = registry
        self.journal = journal or ToolJournal()
        self.heartbeat_every = heartbeat_every
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fde-tool")

    # -------------------------------------------------------------- helpers

    def _run_with_heartbeat(self, fn: Callable[[], T], detail: str) -> T:
        """Run blocking `fn` in a side thread and heartbeat until it returns."""
        fut = self._pool.submit(fn)
        while True:
            try:
                return fut.result(timeout=self.heartbeat_every)
            except TimeoutError:
                activity.heartbeat(detail)

    # -------------------------------------------------------------- activities

    @activity.defn(name="llm_call")
    def llm_call(self, inp: LLMCallInput) -> LLMCallOutput:
        info = activity.info()
        activity.logger.info(
            "llm_call turn=%s attempt=%s", inp.turn, info.attempt
        )

        def call():
            return self.provider.complete(
                system=inp.system,
                messages=inp.messages,
                tools=self.registry.api_schemas(),
                max_tokens=inp.max_tokens,
            )

        resp = self._run_with_heartbeat(call, f"llm turn {inp.turn}")
        return LLMCallOutput(
            text=resp.text,
            tool_calls=[
                {
                    "id": c.id,
                    "name": c.name,
                    "input": c.input,
                    # Decided here, where the registry lives, so the workflow stays free of
                    # registry access and the gate is visible in history alongside the call.
                    "requires_approval": self.registry.requires_approval(c.name),
                }
                for c in resp.tool_calls
            ],
            stop_reason=resp.stop_reason,
            raw_content=resp.raw_content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.provider.model,
        )

    @activity.defn(name="tool_execute")
    def tool_execute(self, inp: ToolCallInput) -> ToolCallOutput:
        info = activity.info()
        cached = self.journal.get(inp.idempotency_key)
        if cached is not None:
            activity.logger.info(
                "tool_execute %s attempt=%s served from journal (key=%s)",
                inp.name,
                info.attempt,
                inp.idempotency_key,
            )
            return cached

        activity.logger.info("tool_execute %s attempt=%s", inp.name, info.attempt)
        t0 = time.perf_counter()
        res = self._run_with_heartbeat(
            lambda: self.registry.execute(inp.name, inp.input), f"tool {inp.name}"
        )
        out = ToolCallOutput(content=res.content, is_error=res.is_error)
        # Record BEFORE returning. If the process dies between this line and Temporal recording
        # the completion, the retry is served from the journal and the tool does not run twice.
        self.journal.put(inp.idempotency_key, out)
        activity.logger.info(
            "tool_execute %s done in %.0f ms", inp.name, (time.perf_counter() - t0) * 1000
        )
        return out
