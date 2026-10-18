"""Runtime configuration, loaded from environment variables and an optional .env file.

Every knob the kit exposes lives here so a new deployment is a config change, not a code change.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FDE_", extra="ignore")

    # LLM
    provider: str = Field(default="anthropic", description="'anthropic' or 'mock'")
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Read from .env or the environment; passed to the SDK explicitly.",
    )
    anthropic_workspace_id: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_WORKSPACE_ID",
        description="Required only for org-level keys not scoped to a workspace (wrkspc_...).",
    )
    model: str = Field(default="claude-sonnet-5-5", description="Model id passed to the provider")
    max_tokens: int = 2048

    # Agent loop
    max_turns: int = Field(default=10, description="Hard cap on LLM round-trips per run")
    system_prompt: str = (
        "You are a careful assistant. Use the provided tools when they help answer "
        "precisely. When you have the answer, reply with it directly and concisely."
    )

    # Tracing
    otel_enabled: bool = True
    otel_endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="OTLP/HTTP endpoint. Default is a local Arize Phoenix instance.",
    )
    otel_service_name: str = "fde-kit"

    # Evals
    eval_pass_threshold: float = Field(
        default=0.85, description="CI fails if pass rate drops below this"
    )


def get_settings() -> Settings:
    return Settings()
