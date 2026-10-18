"""Worker process: hosts the workflow and the activities.

    python -m fde_kit.durable.worker                       # real provider, built-in tools
    python -m fde_kit.durable.worker --provider mock --mock-script path.json
    python -m fde_kit.durable.worker --tools fde_kit.durable.demo_tools:registry

Any number of workers can run against the same task queue. Kill one mid-run and another picks
the run up; that is the whole point, and examples/crash_demo.py shows it happening.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import Settings, get_settings
from ..llm import LLMProvider, MockProvider, make_provider
from ..tools import ToolRegistry
from .activities import AgentActivities
from .journal import ToolJournal
from .workflow import AgentWorkflow

DEFAULT_TASK_QUEUE = "fde-agent"
DEFAULT_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")

log = logging.getLogger("fde.worker")


def load_registry(spec: str | None) -> ToolRegistry:
    """'module.path:attr' -> ToolRegistry. Defaults to the built-in tools."""
    if not spec:
        from ..builtin_tools import registry

        return registry
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr or "registry")


def build_provider(settings: Settings, name: str | None, mock_script: str | None) -> LLMProvider:
    name = name or settings.provider
    if name == "mock":
        script = json.loads(open(mock_script).read()) if mock_script else []
        return MockProvider(script)
    return make_provider(
        name,
        settings.model,
        api_key=settings.anthropic_api_key,
        workspace_id=settings.anthropic_workspace_id,
    )


async def run_worker(
    *,
    address: str = DEFAULT_ADDRESS,
    task_queue: str = DEFAULT_TASK_QUEUE,
    provider: LLMProvider | None = None,
    registry: ToolRegistry | None = None,
    journal_dir: str = ".fde/journal",
    client: Client | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    settings = get_settings()
    client = client or await Client.connect(address)
    provider = provider or build_provider(settings, None, None)
    registry = registry or load_registry(None)
    acts = AgentActivities(provider, registry, ToolJournal(journal_dir))

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[acts.llm_call, acts.tool_execute],
        activity_executor=ThreadPoolExecutor(max_workers=16),
    )
    log.info("worker up: queue=%s pid=%s tools=%s", task_queue, os.getpid(), registry.names())
    if stop is None:
        await worker.run()
    else:
        async with worker:
            await stop.wait()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="fde-kit Temporal worker")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--task-queue", default=DEFAULT_TASK_QUEUE)
    p.add_argument("--provider", choices=["anthropic", "mock"])
    p.add_argument("--mock-script", help="JSON file with a MockProvider script")
    p.add_argument("--tools", help="module.path:attr of a ToolRegistry")
    p.add_argument("--journal-dir", default=".fde/journal")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    provider = build_provider(settings, args.provider, args.mock_script)
    registry = load_registry(args.tools)
    asyncio.run(
        run_worker(
            address=args.address,
            task_queue=args.task_queue,
            provider=provider,
            registry=registry,
            journal_dir=args.journal_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
