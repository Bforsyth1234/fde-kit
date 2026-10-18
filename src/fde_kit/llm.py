"""LLM provider abstraction.

The agent loop only talks to `LLMProvider.complete(...)`, which returns a normalised
`LLMResponse`. Two providers ship:

- `AnthropicProvider` - the default, using the Messages API with client tools.
- `MockProvider` - deterministic, scripted responses for tests and offline eval runs.

Adding another vendor means implementing `complete` and translating its tool-call format into
`ToolCall`. Nothing above this module should know which vendor is in use.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    """Vendor-shaped assistant content, replayed verbatim into the next request."""


class LLMProvider(Protocol):
    model: str

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None, workspace_id: str | None = None):
        from anthropic import Anthropic

        self.model = model
        self.workspace_id = workspace_id
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        # Note: the current Messages API has no sampling-temperature parameter; determinism
        # comes from the eval design (checks tolerant of phrasing), not from the request.
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if self.workspace_id:
            kwargs["workspace_id"] = self.workspace_id
        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
                raw.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            stop_reason=resp.stop_reason or "end_turn",
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            raw_content=raw,
        )


# --------------------------------------------------------------------------- Mock


class MockProvider:
    """Replays a scripted list of responses, one per `complete` call.

    Each script entry is either a plain string (final answer) or a dict:
        {"text": "...", "tool_calls": [{"name": "add", "input": {"a": 1, "b": 2}}]}
    Tool-call ids are generated. If the script runs out, the mock returns a fixed final answer so
    a misbehaving loop terminates instead of hanging.
    """

    model = "mock"

    def __init__(self, script: Iterable[str | dict[str, Any]] = ()):
        self._script = list(script)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        # Snapshot: the agent mutates `messages` in place after this returns.
        self.calls.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": list(tools)}
        )
        if self._i >= len(self._script):
            return LLMResponse(text="[mock: script exhausted]", stop_reason="end_turn")
        entry = self._script[self._i]
        self._i += 1

        if isinstance(entry, str):
            return LLMResponse(
                text=entry, raw_content=[{"type": "text", "text": entry}], usage=Usage(10, 5)
            )

        text = entry.get("text", "")
        calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []
        if text:
            raw.append({"type": "text", "text": text})
        for n, tc in enumerate(entry.get("tool_calls", [])):
            call = ToolCall(id=f"toolu_mock_{self._i}_{n}", name=tc["name"], input=tc["input"])
            calls.append(call)
            raw.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=Usage(10, 5),
            raw_content=raw,
        )


def make_provider(
    name: str, model: str, api_key: str | None = None, workspace_id: str | None = None
) -> LLMProvider:
    if name == "anthropic":
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or export it in your shell."
            )
        return AnthropicProvider(model=model, api_key=api_key, workspace_id=workspace_id)
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider: {name}")
