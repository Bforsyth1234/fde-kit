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
from temporalio.worker import Replayer, Worker

from ..config import Settings, get_settings
from ..llm import LLMProvider, MockProvider, make_provider
from ..tools import ToolRegistry
from .activities import AgentActivities
from .crypto import EncryptionCodec, PayloadCipher, data_converter
from .journal import journal_from_url
from .workflow import AgentWorkflow as _LiveWorkflow

DEFAULT_TASK_QUEUE = "fde-agent"
DEFAULT_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")

log = logging.getLogger("fde.worker")


def payload_cipher(settings: Settings | None = None) -> PayloadCipher | None:
    settings = settings or get_settings()
    return PayloadCipher.from_settings(settings.payload_key, settings.payload_keys)


def cipher_of(client: Client) -> PayloadCipher | None:
    codec = client.data_converter.payload_codec
    return codec.cipher if isinstance(codec, EncryptionCodec) else None


def replayer(workflows: list[type], settings: Settings | None = None) -> Replayer:
    """A `Replayer` that can read encrypted histories. The pre-deploy replay check
    (`tests/test_versioning.py`, README "Upgrading a running agent") must use this, or every
    production history will fail with 'Unknown payload encoding' once a key is in use."""
    return Replayer(workflows=workflows, data_converter=data_converter(payload_cipher(settings)))


async def connect(address: str = DEFAULT_ADDRESS, settings: Settings | None = None) -> Client:
    """Connect with the payload codec, so this process encrypts what it sends and decrypts
    what it reads. Every client and worker in a deployment must use this, not a bare
    `Client.connect`, or they will see ciphertext (and the server would see plaintext)."""
    return await Client.connect(address, data_converter=data_converter(payload_cipher(settings)))


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
        data = json.loads(open(mock_script).read()) if mock_script else []
        if isinstance(data, dict):
            return MockProvider(data.get("script", []), data.get("utility_reply", "[mock]"))
        return MockProvider(data)
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
    journal_dir: str | None = None,
    journal_url: str | None = None,
    client: Client | None = None,
    stop: asyncio.Event | None = None,
    workflow_cls: type | None = None,
) -> None:
    """`workflow_cls` lets a worker serve a specific workflow version (e.g. a frozen v1
    during a rollout, or in the versioning tests). Defaults to the live AgentWorkflow.

    Journal selection, first match wins: `journal_url`, `journal_dir` (file backend),
    `FDE_JOURNAL_URL` from settings, else `.fde/journal` on local disk."""
    settings = get_settings()
    if client is None:
        cipher = payload_cipher(settings)
        client = await connect(address, settings)
    else:
        # One source of truth per process: the journal seals with whatever the client's codec
        # uses, so a client built without the codec gets an unencrypted journal to match, and
        # the two can never disagree about what "encrypted" means.
        cipher = cipher_of(client)
    provider = provider or build_provider(settings, None, None)
    registry = registry or load_registry(None)
    if registry.tenant_id is not None and cipher is None:
        raise RuntimeError(
            f"worker is bound to tenant {registry.tenant_id!r} but no FDE_PAYLOAD_KEY is set; "
            "its runs' prompts and tool results would be stored in plaintext in Temporal's "
            "history and the journal. Set a key (`fde keygen`) or unbind the registry."
        )
    url = journal_url or (f"file:{journal_dir}" if journal_dir else None) or settings.journal_url
    journal = journal_from_url(url, cipher)
    acts = AgentActivities(provider, registry, journal)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_cls or _LiveWorkflow],
        activities=[acts.llm_call, acts.tool_execute, acts.summarize_result],
        activity_executor=ThreadPoolExecutor(max_workers=16),
    )
    log.info(
        "worker up: queue=%s pid=%s tools=%s journal=%r payloads=%s",
        task_queue,
        os.getpid(),
        registry.names(),
        journal,
        "encrypted" if cipher else "PLAINTEXT",
    )
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
    p.add_argument("--journal-dir", help="file-backed journal directory")
    p.add_argument("--journal-url", help="postgresql://... (overrides FDE_JOURNAL_URL)")
    p.add_argument(
        "--workflow-module",
        help="module.path exposing AgentWorkflow to serve (default: fde_kit.durable.workflow)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    provider = build_provider(settings, args.provider, args.mock_script)
    registry = load_registry(args.tools)
    workflow_cls = None
    if args.workflow_module:
        workflow_cls = importlib.import_module(args.workflow_module).AgentWorkflow
    asyncio.run(
        run_worker(
            address=args.address,
            task_queue=args.task_queue,
            provider=provider,
            registry=registry,
            journal_dir=args.journal_dir,
            journal_url=args.journal_url,
            workflow_cls=workflow_cls,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
