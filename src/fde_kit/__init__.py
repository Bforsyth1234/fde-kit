"""fde-kit: a slim starter kit for deploying LLM agents."""

from .agent import Agent, AgentResult, Step
from .config import Settings, get_settings
from .llm import AnthropicProvider, LLMResponse, MockProvider, ToolCall, Usage, make_provider
from .record import (
    ReplayDiverged,
    ReplayProvider,
    ReplayRegistry,
    ReplaySession,
    RunRecord,
    RunRecorder,
    default_runs_dir,
    fork_session,
    live_llm_session,
    new_run_id,
    replay_session,
    write_record_from_result,
)
from .tools import Source, ToolRegistry, ToolResult, ToolSpec
from .tracing import init_tracing

__all__ = [
    "Agent",
    "AgentResult",
    "AnthropicProvider",
    "LLMResponse",
    "MockProvider",
    "ReplayDiverged",
    "ReplayProvider",
    "ReplayRegistry",
    "ReplaySession",
    "RunRecord",
    "RunRecorder",
    "Settings",
    "Source",
    "Step",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "default_runs_dir",
    "fork_session",
    "get_settings",
    "init_tracing",
    "live_llm_session",
    "make_provider",
    "new_run_id",
    "replay_session",
    "write_record_from_result",
]
