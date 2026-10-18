"""Durable runtime tests against a real local Temporal dev server.

`WorkflowEnvironment.start_local()` downloads and starts the Temporal CLI dev server on first
use. If that can't happen (no network, blocked download), these tests skip rather than fail.

The crash test is the one that matters: it runs the worker as a subprocess, SIGKILLs it while a
tool activity is executing, starts a fresh worker, and asserts the workflow finishes without
re-running the LLM call that had already completed.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

temporalio = pytest.importorskip("temporalio")
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from fde_kit.builtin_tools import registry  # noqa: E402
from fde_kit.durable.client import history_summary, start_run, status  # noqa: E402
from fde_kit.durable.journal import ToolJournal  # noqa: E402
from fde_kit.durable.models import ToolCallOutput  # noqa: E402
from fde_kit.durable.worker import run_worker  # noqa: E402
from fde_kit.llm import MockProvider  # noqa: E402

pytestmark = pytest.mark.durable


@pytest.fixture(scope="module")
def env():
    async def _start():
        try:
            return await asyncio.wait_for(WorkflowEnvironment.start_local(), timeout=120)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"local Temporal dev server unavailable: {exc}")

    loop = asyncio.new_event_loop()
    e = loop.run_until_complete(_start())
    yield e, loop
    loop.run_until_complete(e.shutdown())
    loop.close()


TWO_TOOL_SCRIPT = [
    {"tool_calls": [{"name": "lookup_policy", "input": {"topic": "refund policy"}}]},
    {"tool_calls": [{"name": "calculator", "input": {"expression": "30 * 2"}}]},
    "Doubled, the refund window would be 60 days.",
]


def test_workflow_mirrors_in_process_agent(env):
    e, loop = env

    async def go():
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-basic",
                provider=MockProvider(TWO_TOOL_SCRIPT),
                registry=registry,
                journal_dir=tempfile.mkdtemp(),
                stop=stop,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("refund doubled?", client=e.client, task_queue="t-basic")
        out = await h.result()
        hist = await history_summary(h)
        stop.set()
        await worker
        return out, hist

    out, hist = loop.run_until_complete(go())
    assert out.output == "Doubled, the refund window would be 60 days."
    assert out.tools_called == ["lookup_policy", "calculator"]
    assert out.stopped_reason == "end_turn" and out.turns == 3
    assert out.steps[1].tool_results[0]["content"] == "60"
    names = [a["activity"] for a in hist["activities"]]
    assert names == ["llm_call", "tool_execute", "llm_call", "tool_execute", "llm_call"]
    assert all(a["attempts"] == 1 and a["completed"] for a in hist["activities"])


def test_max_turns_is_a_result_not_an_error(env):
    e, loop = env

    async def go():
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-max",
                provider=MockProvider(
                    [{"tool_calls": [{"name": "current_date", "input": {}}]}] * 9
                ),
                registry=registry,
                journal_dir=tempfile.mkdtemp(),
                stop=stop,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("loop", client=e.client, task_queue="t-max", max_turns=2)
        out = await h.result()
        stop.set()
        await worker
        return out

    out = loop.run_until_complete(go())
    assert out.stopped_reason == "max_turns" and out.turns == 2


def test_journal_hit_skips_execution(env, tmp_path: Path):
    """A pre-recorded result under the idempotency key is returned without running the tool."""
    e, loop = env
    journal = ToolJournal(tmp_path)
    wf_id = "agent-journal-test"
    # The mock's first tool_use id is deterministic: toolu_mock_<turn-1>_<n>.
    journal.put(f"{wf_id}:1:toolu_mock_0_0", ToolCallOutput(content="FROM_JOURNAL"))

    async def go():
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-journal",
                provider=MockProvider(
                    [
                        {"tool_calls": [{"name": "calculator", "input": {"expression": "1+1"}}]},
                        "done",
                    ]
                ),
                registry=registry,
                journal_dir=str(tmp_path),
                stop=stop,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("x", client=e.client, task_queue="t-journal", workflow_id=wf_id)
        out = await h.result()
        stop.set()
        await worker
        return out

    out = loop.run_until_complete(go())
    res = out.steps[0].tool_results[0]
    assert res["content"] == "FROM_JOURNAL" and res["from_journal"] is True


def _spawn_worker(address: str, queue: str, script_path: str, journal_dir: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "fde_kit.durable.worker",
        "--address",
        address,
        "--task-queue",
        queue,
        "--provider",
        "mock",
        "--mock-script",
        script_path,
        "--tools",
        "fde_kit.durable.demo_tools:registry",
        "--journal-dir",
        journal_dir,
    ]
    env = {**os.environ, "FDE_OTEL_ENABLED": "false"}
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_sigkill_worker_mid_tool_then_resume(env, tmp_path: Path):
    e, loop = env
    address = e.client.service_client.config.target_host
    script = [
        {"tool_calls": [{"name": "wait_seconds", "input": {"seconds": 6}}]},
        {"tool_calls": [{"name": "calculator", "input": {"expression": "17 * 23"}}]},
        "17 * 23 = 391.",
    ]
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))
    journal_dir = str(tmp_path / "journal")

    async def go():
        w1 = _spawn_worker(address, "t-crash", str(script_path), journal_dir)
        try:
            await asyncio.sleep(1.5)
            h = await start_run("wait then multiply", client=e.client, task_queue="t-crash")

            # Wait until turn-1 LLM call is done and the slow tool is executing.
            deadline = time.time() + 30
            while time.time() < deadline:
                st = await status(h)
                if st.phase == "tools" and st.current_tool == "wait_seconds":
                    break
                await asyncio.sleep(0.2)
            else:
                raise AssertionError("never reached tool phase")
            await asyncio.sleep(1.0)  # make sure the activity has started on the worker
            before = await history_summary(h)

            w1.send_signal(signal.SIGKILL)
            w1.wait(timeout=10)

            # No worker: the workflow must still be open, not failed.
            desc = await h.describe()
            assert desc.status.name == "RUNNING"

            w2 = _spawn_worker(address, "t-crash", str(script_path), journal_dir)
            try:
                out = await asyncio.wait_for(h.result(), timeout=90)
                after = await history_summary(h)
            finally:
                w2.send_signal(signal.SIGTERM)
                w2.wait(timeout=10)
            return before, out, after
        finally:
            if w1.poll() is None:
                w1.kill()

    before, out, after = loop.run_until_complete(go())

    assert out.output == "17 * 23 = 391."
    assert out.tools_called == ["wait_seconds", "calculator"]

    acts = after["activities"]
    # Turn-1 LLM call completed once before the kill and was never re-run.
    first_llm = acts[0]
    assert first_llm["activity"] == "llm_call" and first_llm["attempts"] == 1
    # The slow tool was in flight during the kill and needed a second attempt.
    slow = acts[1]
    assert slow["activity"] == "tool_execute" and slow["attempts"] >= 2 and slow["completed"]
    # Everything after the restart ran exactly once.
    assert all(a["attempts"] == 1 and a["completed"] for a in acts[2:])
    # And the kill happened before the slow tool completed.
    assert not before["activities"][1]["completed"]
