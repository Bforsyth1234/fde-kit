"""The agent loop.

Deliberately small. One class, one public method. Each run:

    prompt -> [LLM call -> tool calls -> tool results]* -> final text

Every LLM call and every tool execution is its own span, nested under one run span, so a trace
in Phoenix reads like the loop itself. The loop stops when the model returns no tool calls, or
when `max_turns` is hit (recorded as `stopped_reason="max_turns"` rather than raised, so callers
and evals can inspect what happened).

`fde_kit.durable.workflow.AgentWorkflow` is this same loop with the two side effects moved into
Temporal activities. Read them side by side; keeping this one plain is what makes that possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .citations import CitationLedger, cited, unresolved_citations
from .config import Settings, get_settings
from .llm import LLMProvider, Usage, make_provider
from .prompts import PromptRef, load_prompt
from .record import (
    ReplayDiverged,
    ReplaySession,
    RunRecorder,
    llm_row,
    new_run_id,
    result_row,
    tool_row,
    user_row,
    utc_now,
)
from .tools import Source, ToolRegistry
from .tracing import get_tracer, tenant_scope, trace_content


@dataclass
class Step:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    """Per tool call: {name, content, raw_content, is_error, sources: [Source dicts],
    citations: [n, ...]}. `content` is what the model saw (run-wide `[n]` numbers);
    `raw_content` is the tool's own output; `citations` are the run-wide numbers assigned to
    that call's sources."""
    latency_ms: float
    stop_reason: str = ""
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    """The vendor-shaped assistant content for this turn, as recorded in a run file."""
    usage: Usage = field(default_factory=Usage)


@dataclass
class AgentResult:
    output: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_reason: str = "end_turn"
    """"end_turn" | "max_turns" | "replay_diverged"."""
    total_ms: float = 0.0
    sources: list[Source] = field(default_factory=list)
    """Every Source any tool returned during the run, in citation order: `sources[n-1]` is [n]."""
    model: str = ""
    prompt_name: str = ""
    prompt_sha256: str = ""
    """Which model and which prompt produced this output. See fde_kit.prompts."""
    prompt: str = ""
    """The opening user prompt."""
    run_id: str = ""
    """Set when the run was recorded. See fde_kit.record."""
    record_path: str = ""
    diverged_at: int | None = None
    divergence: dict[str, Any] | None = None
    """On `stopped_reason="replay_diverged"`: the turn and {"expected", "got"} tool calls."""

    @property
    def tools_called(self) -> list[str]:
        return [tc["name"] for s in self.steps for tc in s.tool_calls]

    @property
    def turns(self) -> int:
        return len(self.steps)

    @property
    def cited(self) -> list[Source]:
        """The subset of `sources` the final answer cites by number."""
        return cited(self.output, self.sources)

    @property
    def fabricated_citations(self) -> list[int]:
        """Citation numbers in the answer that no tool returned. Should always be empty."""
        return unresolved_citations(self.output, self.sources)


def resolve_prompt(settings: Settings, system_prompt: str | None = None) -> PromptRef:
    """Explicit text wins, then FDE_SYSTEM_PROMPT, then the named prompt file."""
    if system_prompt is not None:
        return PromptRef.inline(system_prompt)
    if settings.system_prompt:
        return PromptRef.inline(settings.system_prompt)
    return load_prompt(settings.prompt, settings.prompts_dir)


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
        system_prompt: str | None = None,
        recorder: RunRecorder | None = None,
    ):
        self.settings = settings or get_settings()
        self.registry = registry
        self.provider = provider or make_provider(
            self.settings.provider,
            self.settings.model,
            api_key=self.settings.anthropic_api_key,
            workspace_id=self.settings.anthropic_workspace_id,
        )
        self.prompt: PromptRef = resolve_prompt(self.settings, system_prompt)
        self.system_prompt = self.prompt.text
        self.recorder = recorder
        """An open RunRecorder every run of this agent writes to. Usually None: pass
        `record=path` to `run` instead, which opens one per run."""
        self._tracer = get_tracer()

    def run(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        session: ReplaySession | None = None,
        record: str | Path | None = None,
    ) -> AgentResult:
        """Run the loop on `prompt`.

        `session` (see `fde_kit.record`) swaps in a replaying provider/registry for this run and
        may inject a user message before a given turn (a fork). `record` opens a `RunRecorder`
        at that path and streams every row of this run to it; replaying and recording in one
        run is how a fork writes its own file while reading its parent's.
        """
        max_turns = max_turns or self.settings.max_turns
        provider = session.provider if session is not None else self.provider
        registry = session.registry if session is not None else self.registry
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result = AgentResult(
            output="",
            model=provider.model,
            prompt_name=self.prompt.name,
            prompt_sha256=self.prompt.sha256,
            prompt=prompt,
        )
        ledger = CitationLedger()
        t0 = time.perf_counter()

        recorder = self.recorder
        if record is not None:
            recorder = RunRecorder(record, self.build_header(prompt, max_turns, session))
        if recorder is not None:
            result.run_id = recorder.run_id
            result.record_path = str(recorder.path)
        rec = _Rec(recorder)

        with (
            tenant_scope(registry.tenant_id),
            self._tracer.start_as_current_span("agent.run") as run_span,
            rec,
        ):
            run_span.set_attribute("input.value", trace_content(prompt))
            run_span.set_attribute("llm.model_name", provider.model)
            run_span.set_attribute("llm.prompt_name", self.prompt.name)
            run_span.set_attribute("llm.prompt_sha256", self.prompt.sha256)
            run_span.set_attribute("agent.max_turns", max_turns)
            run_span.set_attribute("agent.replay", session is not None)
            if result.run_id:
                run_span.set_attribute("agent.run_id", result.run_id)
            if session is not None and session.parent_run_id:
                run_span.set_attribute("agent.parent_run_id", session.parent_run_id)
            rec.write(user_row(1, prompt))

            for turn in range(1, max_turns + 1):
                if session is not None and session.inject and session.inject[0] == turn:
                    text = session.inject[1]
                    messages.append({"role": "user", "content": text})
                    rec.write(user_row(turn, text))
                set_turn = getattr(registry, "set_turn", None)
                if callable(set_turn):
                    set_turn(turn)
                try:
                    step, final_text, usage = self._step(
                        turn, messages, ledger, provider=provider, registry=registry, rec=rec
                    )
                except ReplayDiverged as exc:
                    result.stopped_reason = "replay_diverged"
                    result.diverged_at = exc.turn
                    result.divergence = {"expected": exc.expected, "got": exc.got}
                    result.output = result.steps[-1].assistant_text if result.steps else ""
                    run_span.set_attribute("agent.diverged_at", exc.turn)
                    break
                result.steps.append(step)
                result.usage = result.usage + usage
                if final_text is not None:
                    result.output = final_text
                    result.stopped_reason = "end_turn"
                    break
            else:
                result.stopped_reason = "max_turns"
                result.output = result.steps[-1].assistant_text if result.steps else ""

            result.total_ms = (time.perf_counter() - t0) * 1000
            result.sources = list(ledger.sources)
            run_span.set_attribute("output.value", trace_content(result.output))
            run_span.set_attribute("agent.turns", result.turns)
            run_span.set_attribute("agent.stopped_reason", result.stopped_reason)
            run_span.set_attribute("agent.sources", len(result.sources))
            run_span.set_attribute("agent.cited", len(result.cited))
            run_span.set_attribute("agent.fabricated_citations", len(result.fabricated_citations))
            run_span.set_attribute("llm.token_count.prompt", result.usage.input_tokens)
            run_span.set_attribute("llm.token_count.completion", result.usage.output_tokens)
            rec.close(result_row(result))
        return result

    def build_header(
        self, prompt: str, max_turns: int | None = None, session: ReplaySession | None = None
    ) -> dict[str, Any]:
        """The `header` row for a run of this agent (see the record format in fde_kit.record)."""
        provider = session.provider if session is not None else self.provider
        registry = session.registry if session is not None else self.registry
        return {
            "kind": "header",
            "version": 1,
            "run_id": new_run_id(),
            "created_at": utc_now(),
            "runtime": "agent",
            "tenant_id": registry.tenant_id,
            "model": provider.model,
            "prompt_name": self.prompt.name,
            "prompt_sha256": self.prompt.sha256,
            "prompt_text": self.prompt.text,
            "max_turns": max_turns or self.settings.max_turns,
            "max_tokens": self.settings.max_tokens,
            "tools": registry.names(),
            "parent_run_id": session.parent_run_id if session is not None else None,
            "fork_at": session.fork_at if session is not None else None,
            "workflow_id": None,
        }

    # ------------------------------------------------------------------ internals

    def _step(
        self,
        turn: int,
        messages: list[dict[str, Any]],
        ledger: CitationLedger | None = None,
        *,
        provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        rec: _Rec | None = None,
    ) -> tuple[Step, str | None, Usage]:
        """One LLM round-trip plus any tool executions it requested.

        Returns (step, final_text, usage). final_text is None while the loop should continue.
        """
        t0 = time.perf_counter()
        ledger = ledger if ledger is not None else CitationLedger()
        provider = provider or self.provider
        registry = registry or self.registry
        rec = rec or _Rec(None)
        with self._tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.turn", turn)
            span.set_attribute("llm.prompt_sha256", self.prompt.sha256)
            span.set_attribute("input.value", trace_content(messages[-1]))
            resp = provider.complete(
                system=self.system_prompt,
                messages=messages,
                tools=registry.api_schemas(),
                max_tokens=self.settings.max_tokens,
            )
            span.set_attribute("output.value", trace_content(resp.raw_content))
            span.set_attribute("llm.stop_reason", resp.stop_reason)
            span.set_attribute("llm.token_count.prompt", resp.usage.input_tokens)
            span.set_attribute("llm.token_count.completion", resp.usage.output_tokens)
        rec.write(llm_row(turn, resp))

        messages.append({"role": "assistant", "content": resp.raw_content})

        if not resp.tool_calls:
            step = Step(
                turn,
                resp.text,
                [],
                [],
                (time.perf_counter() - t0) * 1000,
                stop_reason=resp.stop_reason,
                raw_content=resp.raw_content,
                usage=resp.usage,
            )
            return step, resp.text, resp.usage

        tool_result_blocks: list[dict[str, Any]] = []
        recorded_results: list[dict[str, Any]] = []
        for call in resp.tool_calls:
            with self._tracer.start_as_current_span(f"tool.{call.name}") as tspan:
                tspan.set_attribute("tool.name", call.name)
                tspan.set_attribute("input.value", trace_content(call.input))
                res = registry.execute(call.name, call.input)
                rec.write(tool_row(turn, call, res))
                content, numbers = ledger.add(res.content, res.sources)
                tspan.set_attribute("output.value", trace_content(content))
                tspan.set_attribute("tool.is_error", res.is_error)
                if res.sources:
                    # Ids only, never content: safe on a shared trace store.
                    tspan.set_attribute("tool.source_ids", [s.source_id for s in res.sources])
                    tspan.set_attribute("tool.citations", numbers)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
            }
            if res.is_error:
                block["is_error"] = True
            tool_result_blocks.append(block)
            recorded_results.append(
                {
                    "name": call.name,
                    "content": content,
                    "raw_content": res.content,
                    "is_error": res.is_error,
                    "sources": [s.to_dict() for s in res.sources],
                    "citations": numbers,
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})
        step = Step(
            turn,
            resp.text,
            [{"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls],
            recorded_results,
            (time.perf_counter() - t0) * 1000,
            stop_reason=resp.stop_reason,
            raw_content=resp.raw_content,
            usage=resp.usage,
        )
        return step, None, resp.usage


class _Rec:
    """A RunRecorder or nothing, with the same surface, so the loop has no `if recorder` noise.

    Used as a context manager: on an exception inside the run it writes the error result row.
    Closing is idempotent, so a normal exit after `close(result_row)` writes nothing more.
    """

    def __init__(self, recorder: RunRecorder | None):
        self.recorder = recorder

    def write(self, row: dict[str, Any]) -> None:
        if self.recorder is not None:
            self.recorder.write(row)

    def close(self, row: dict[str, Any]) -> None:
        if self.recorder is not None:
            self.recorder.close(row)

    def __enter__(self) -> _Rec:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.recorder is not None:
            self.recorder.__exit__(exc_type, exc, tb)
