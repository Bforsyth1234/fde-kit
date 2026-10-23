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
    prompt: str = Field(
        default="default",
        description=(
            "Name of the system prompt (prompts/<name>.md, else the packaged one) or a path to "
            "a .md file. Its name and sha256 are recorded on every run."
        ),
    )
    prompts_dir: str | None = Field(
        default=None, description="Directory of prompt files; default ./prompts if it exists"
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Raw system prompt text. Overrides `prompt` when set; recorded as prompt "
            "'inline' with the hash of the text."
        ),
    )

    # Record / replay
    runs_dir: str = Field(
        default="runs",
        description=(
            "Where recorded runs are written (runs/<run_id>.jsonl). Files are unredacted "
            "local evidence; keep the directory out of version control."
        ),
    )
    record_runs: bool = Field(
        default=False,
        description="Record every `fde run` to runs_dir. `fde rag ask` always records.",
    )

    # Tracing
    otel_enabled: bool = True
    otel_endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="OTLP/HTTP endpoint. Default is a local Arize Phoenix instance.",
    )
    otel_service_name: str = "fde-kit"
    otel_tenant_content: bool = Field(
        default=False,
        description=(
            "Record prompts, tool I/O and answers on spans for tenant-scoped runs. Off by "
            "default: a shared trace store would show every tenant's data to anyone with access."
        ),
    )

    # Durable runtime
    journal_url: str | None = Field(
        default=None,
        description=(
            "Tool idempotency journal: postgresql://user:pw@host/db for a shared journal, "
            "file:<dir> for local files. Default: .fde/journal on local disk."
        ),
    )

    # Multi-tenant retrieval
    rag_url: str | None = Field(
        default=None,
        description=(
            "App connection for retrieval: postgresql://fde_rag_app:...@host/db. Must not be "
            "a superuser, BYPASSRLS, or table owner; the store checks and refuses."
        ),
    )
    rag_admin_url: str = Field(
        default="postgresql://fde:fde@localhost:5432/fde",
        description="Used only by `fde rag migrate` / `seed-demo` to create roles and tables.",
    )
    rag_partition_key: str = Field(
        default="",
        description=(
            "Optional secret. Partition names become an HMAC of the tenant id, so the catalog "
            "doesn't reveal tenant ids to anyone with a candidate list. Set it before the first "
            "migrate; it can't be changed afterwards without renaming partitions."
        ),
    )
    rag_embedder: str = Field(
        default="bge", description="'bge' (BAAI/bge-small-en-v1.5 via fastembed) or 'hash'"
    )

    # Source adapters (PR 4)
    edgar_user_agent: str = Field(
        default="",
        description=(
            "Contact SEC requires in the User-Agent for sec.gov requests, e.g. "
            "'my-app you@example.com'. Required online; never logged or written to disk."
        ),
    )
    edgar_cache_dir: str = Field(
        default=".fde/edgar-cache", description="On-disk cache of sec.gov GET responses"
    )

    # Evals
    eval_pass_threshold: float = Field(
        default=0.85, description="CI fails if pass rate drops below this"
    )


def get_settings() -> Settings:
    return Settings()
