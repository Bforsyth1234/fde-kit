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
from typing import Any

from .citations import CitationLedger, cited, unresolved_citations
from .config import Settings, get_settings
from .llm import LLMProvider, Usage, make_provider
from .prompts import PromptRef, load_prompt
from .tools import Source, ToolRegistry
from .tracing import get_tracer, tenant_scope, trace_content


@dataclass
class Step:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    """Per tool call: {name, content, is_error, sources: [Source dicts], citations: [n, ...]}.
    `citations` are the run-wide numbers assigned to that call's sources."""
    latency_ms: float


@dataclass
class AgentResult:
    output: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_reason: str = "end_turn"
    total_ms: float = 0.0
    sources: list[Source] = field(default_factory=list)
    """Every Source any tool returned during the run, in citation order: `sources[n-1]` is [n]."""
    model: str = ""
    prompt_name: str = ""
    prompt_sha256: str = ""
    """Which model and which prompt produced this output. See fde_kit.prompts."""

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
        self._tracer = get_tracer()

    def run(self, prompt: str, *, max_turns: int | None = None) -> AgentResult:
        max_turns = max_turns or self.settings.max_turns
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result = AgentResult(
            output="",
            model=self.provider.model,
            prompt_name=self.prompt.name,
            prompt_sha256=self.prompt.sha256,
        )
        ledger = CitationLedger()
        t0 = time.perf_counter()

        with (
            tenant_scope(self.registry.tenant_id),
            self._tracer.start_as_current_span("agent.run") as run_span,
        ):
            run_span.set_attribute("input.value", trace_content(prompt))
            run_span.set_attribute("llm.model_name", self.provider.model)
            run_span.set_attribute("llm.prompt_name", self.prompt.name)
            run_span.set_attribute("llm.prompt_sha256", self.prompt.sha256)
            run_span.set_attribute("agent.max_turns", max_turns)

            for turn in range(1, max_turns + 1):
                step, final_text, usage = self._step(turn, messages, ledger)
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
        return result

    # ------------------------------------------------------------------ internals

    def _step(
        self, turn: int, messages: list[dict[str, Any]], ledger: CitationLedger | None = None
    ) -> tuple[Step, str | None, Usage]:
        """One LLM round-trip plus any tool executions it requested.

        Returns (step, final_text, usage). final_text is None while the loop should continue.
        """
        t0 = time.perf_counter()
        ledger = ledger if ledger is not None else CitationLedger()
        with self._tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.turn", turn)
            span.set_attribute("llm.prompt_sha256", self.prompt.sha256)
            span.set_attribute("input.value", trace_content(messages[-1]))
            resp = self.provider.complete(
                system=self.system_prompt,
                messages=messages,
                tools=self.registry.api_schemas(),
                max_tokens=self.settings.max_tokens,
            )
            span.set_attribute("output.value", trace_content(resp.raw_content))
            span.set_attribute("llm.stop_reason", resp.stop_reason)
            span.set_attribute("llm.token_count.prompt", resp.usage.input_tokens)
            span.set_attribute("llm.token_count.completion", resp.usage.output_tokens)

        messages.append({"role": "assistant", "content": resp.raw_content})

        if not resp.tool_calls:
            step = Step(turn, resp.text, [], [], (time.perf_counter() - t0) * 1000)
            return step, resp.text, resp.usage

        tool_result_blocks: list[dict[str, Any]] = []
        recorded_results: list[dict[str, Any]] = []
        for call in resp.tool_calls:
            with self._tracer.start_as_current_span(f"tool.{call.name}") as tspan:
                tspan.set_attribute("tool.name", call.name)
                tspan.set_attribute("input.value", trace_content(call.input))
                res = self.registry.execute(call.name, call.input)
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
        )
        return step, None, resp.usage
