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
    tool_calls: list[dict[str, Any]]  # [{"id", "name", "input", "requires_approval"}]
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
class SummarizeInput:
    tool_name: str
    content: str
    max_chars: int


@dataclass
class SummarizeOutput:
    content: str
    original_chars: int
    summarized: bool


@dataclass
class AgentRunInput:
    prompt: str
    max_turns: int = 10
    system_prompt: str | None = None
    max_tokens: int = 2048
    approval_timeout_seconds: int = 7 * 24 * 3600
    """How long a gated tool call waits for a human before it is treated as rejected."""
    require_approval_for: list[str] = field(default_factory=list)
    """Extra tool names to gate for this run, on top of tools registered with
    requires_approval=True."""


@dataclass
class PendingApproval:
    tool_use_id: str
    name: str
    input: dict[str, Any]
    turn: int


@dataclass
class ApprovalDecision:
    tool_use_id: str
    approved: bool
    by: str = ""
    note: str = ""


@dataclass
class StepRecord:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    """Decisions for gated tool calls in this step: {tool_use_id, name, outcome, by, note}"""
    code_version: int = 1
    """Which version of the workflow code produced this step. Changes mid-run on an upgrade."""


@dataclass
class AgentRunOutput:
    output: str
    steps: list[StepRecord]
    stopped_reason: str
    input_tokens: int
    output_tokens: int
    code_version: int = 1
    """Version of the workflow code that completed the run."""

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
    phase: str  # "llm" | "tools" | "awaiting_approval" | "done"
    tools_called: list[str]
    current_tool: str | None = None
    pending_approval: PendingApproval | None = None
