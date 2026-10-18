"""Plain dataclasses that cross the workflow/activity boundary.

Everything here is JSON-serialisable by Temporal's default data converter. Keep these flat and
boring: they are written into workflow history, so every field costs storage and shows up in
the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMCallInput:
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int
    turn: int


@dataclass
class LLMCallOutput:
    text: str
    tool_calls: list[dict[str, Any]]  # [{"id", "name", "input"}]
    stop_reason: str
    raw_content: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class ToolCallInput:
    name: str
    input: dict[str, Any]
    idempotency_key: str
    tool_use_id: str


@dataclass
class ToolCallOutput:
    content: str
    is_error: bool = False
    from_journal: bool = False  # True when served from the idempotency journal


@dataclass
class AgentRunInput:
    prompt: str
    max_turns: int = 10
    system_prompt: str | None = None
    max_tokens: int = 2048


@dataclass
class StepRecord:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentRunOutput:
    output: str
    steps: list[StepRecord]
    stopped_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def turns(self) -> int:
        return len(self.steps)

    @property
    def tools_called(self) -> list[str]:
        return [tc["name"] for s in self.steps for tc in s.tool_calls]


@dataclass
class RunStatus:
    """Returned by the workflow's `status` query while a run is in flight."""

    turn: int
    phase: str  # "llm" | "tools" | "done"
    tools_called: list[str]
    current_tool: str | None = None
