"""fde-kit: a slim starter kit for deploying LLM agents."""

from .agent import Agent, AgentResult, Step
from .config import Settings, get_settings
from .llm import AnthropicProvider, LLMResponse, MockProvider, ToolCall, Usage, make_provider
from .tools import Source, ToolRegistry, ToolResult, ToolSpec
from .tracing import init_tracing

__all__ = [
    "Agent",
    "AgentResult",
    "AnthropicProvider",
    "LLMResponse",
    "MockProvider",
    "Settings",
    "Source",
    "Step",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "get_settings",
    "init_tracing",
    "make_provider",
]
