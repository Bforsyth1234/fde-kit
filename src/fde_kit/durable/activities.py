"""Activities: the two side-effecting steps of the agent loop.

Activities are where non-determinism is allowed. The LLM call and the tool execution both live
here; the workflow only decides what to call next. Both activities heartbeat while they run, so
if the worker process dies mid-step Temporal notices within `heartbeat_timeout` and reschedules
the activity on another worker instead of waiting for the full `start_to_close_timeout`.

Tool execution consults the idempotency journal first. See journal.py for why.
"""

from __future__ import annotations

import contextvars
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from temporalio import activity

from ..llm import LLMProvider
from ..tools import ToolRegistry
from ..tracing import tenant_scope
from .journal import FileJournal, ToolJournal
from .models import (
    LLMCallInput,
    LLMCallOutput,
    SummarizeInput,
    SummarizeOutput,
    ToolCallInput,
    ToolCallOutput,
)

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
        self.journal = journal or FileJournal()
        self.heartbeat_every = heartbeat_every
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fde-tool")

    # -------------------------------------------------------------- helpers

    def _effective_tenant(self, requested: str | None) -> str | None:
        if (
            requested is not None
            and self.registry.tenant_id is not None
            and requested != self.registry.tenant_id
        ):
            raise PermissionError(
                f"activity for tenant {requested!r} routed to worker bound to "
                f"{self.registry.tenant_id!r}"
            )
        return requested or self.registry.tenant_id

    def _run_with_heartbeat(self, fn: Callable[[], T], detail: str) -> T:
        """Run blocking `fn` in a side thread, preserving ContextVars (`tenant_scope`), and
        heartbeat until it returns."""
        ctx = contextvars.copy_context()
        fut = self._pool.submit(ctx.run, fn)
        while True:
            try:
                return fut.result(timeout=self.heartbeat_every)
            except TimeoutError:
                activity.heartbeat(detail)

    # -------------------------------------------------------------- activities

    @activity.defn(name="llm_call")
    def llm_call(self, inp: LLMCallInput) -> LLMCallOutput:
        tid = self._effective_tenant(inp.tenant_id)
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

        with tenant_scope(tid):
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
        tid = self._effective_tenant(inp.tenant_id)
        info = activity.info()
        cached = self.journal.get(inp.idempotency_key, tenant_id=tid, tool_name=inp.name)
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
        with tenant_scope(tid):
            res = self._run_with_heartbeat(
                lambda: self.registry.execute(inp.name, inp.input), f"tool {inp.name}"
            )
        out = ToolCallOutput(content=res.content, is_error=res.is_error)
        # Record BEFORE returning. If the process dies between this line and Temporal recording
        # the completion, the retry is served from the journal and the tool does not run twice.
        self.journal.put(inp.idempotency_key, out, tool_name=inp.name, tenant_id=tid)
        # Test hook: hold the activity open after journaling so a test can kill the worker in
        # exactly the window the journal exists for. Never set in production.
        hang = float(os.environ.get("FDE_TEST_HANG_AFTER_JOURNAL", "0") or 0)
        if hang > 0:
            self._run_with_heartbeat(lambda: time.sleep(hang), "test hang")
        activity.logger.info(
            "tool_execute %s done in %.0f ms", inp.name, (time.perf_counter() - t0) * 1000
        )
        return out

    @activity.defn(name="summarize_result")
    def summarize_result(self, inp: SummarizeInput) -> SummarizeOutput:
        """v2: compress a long tool result before it goes back into the model's context.

        Added behind `workflow.patched("v2-summarize-results")`. Short results pass through
        untouched so the common case costs nothing.
        """
        tid = self._effective_tenant(inp.tenant_id)
        n = len(inp.content)
        if n <= inp.max_chars:
            return SummarizeOutput(content=inp.content, original_chars=n, summarized=False)

        def call():
            return self.provider.complete(
                system=(
                    "Condense the following tool output to the facts a downstream assistant "
                    "needs, under the requested length. Keep numbers, names and errors verbatim."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Tool: {inp.tool_name}\nMax chars: {inp.max_chars}\n\n{inp.content}"
                        ),
                    }
                ],
                tools=[],
                max_tokens=max(64, inp.max_chars // 3),
            )

        with tenant_scope(tid):
            resp = self._run_with_heartbeat(call, f"summarize {inp.tool_name}")
        text = resp.text.strip() or inp.content[: inp.max_chars]
        return SummarizeOutput(content=text, original_chars=n, summarized=True)

