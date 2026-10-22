"""Upgrading the workflow code while runs are in flight.

What `workflow.patched` actually guarantees (and what these tests pin down):

1. A run started on v1 finishes on v2 **as a v1 run**. The first time v2 code replays a v1
   history it evaluates the patch during replay, finds no marker, and takes the old branch; from
   then on that run is pinned to the old branch for its lifetime, so its behaviour never changes
   mid-conversation. No non-determinism error, no summarizer activity, code_version=1 on every
   step. For an agent this is the behaviour you want: the model's context doesn't change shape
   halfway through a run because someone deployed.

2. Every v1 history on record replays cleanly through the v2 code. `Replayer` checks exactly
   that, and it is the test to run in CI before any deploy that touches a workflow.

3. A run that starts on v2 takes the new branch from its first step and records the patch marker.

`tests/fixtures/workflow_v1.py` is a frozen copy of the workflow at the commit before the
patch. Do not edit it; that is the point.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

temporalio = pytest.importorskip("temporalio")
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from fde_kit.durable.client import history_summary, start_run, status  # noqa: E402
from fde_kit.durable.worker import replayer, run_worker  # noqa: E402
from fde_kit.durable.workflow import CODE_VERSION, AgentWorkflow  # noqa: E402
from fde_kit.llm import MockProvider  # noqa: E402
from tests.fixtures.workflow_v1 import AgentWorkflow as AgentWorkflowV1  # noqa: E402

pytestmark = pytest.mark.durable

# Two long-output tool calls separated by a slow one. The slow one is where the worker swap
# happens, so turn 1 is served by v1 and turn 3 by v2.
SCRIPT = {
    "script": [
        {"tool_calls": [{"name": "fetch_report", "input": {"report_id": "Q1"}}]},
        {"tool_calls": [{"name": "wait_seconds", "input": {"seconds": 6}}]},
        {"tool_calls": [{"name": "fetch_report", "input": {"report_id": "Q2"}}]},
        "Both reports reviewed.",
    ],
    "utility_reply": "SUMMARY: all metrics nominal.",
}


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


def _spawn(
    address: str, queue: str, script_path: str, journal: str, module: str
) -> subprocess.Popen:
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
        journal,
        "--workflow-module",
        module,
    ]
    env = {**os.environ, "FDE_OTEL_ENABLED": "false", "PYTHONPATH": os.getcwd()}
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _wait_for_tool(h, name: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = await status(h)
        if st.phase == "tools" and st.current_tool == name:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"never saw tool {name!r} in flight")


def test_run_started_on_v1_finishes_on_v2(env, tmp_path: Path):
    e, loop = env
    address = e.client.service_client.config.target_host
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(SCRIPT))
    journal = str(tmp_path / "journal")

    async def go():
        v1 = _spawn(address, "t-upgrade", str(script_path), journal, "tests.fixtures.workflow_v1")
        try:
            await asyncio.sleep(1.5)
            h = await start_run("review Q1 and Q2", client=e.client, task_queue="t-upgrade")
            await _wait_for_tool(h, "wait_seconds")
            await asyncio.sleep(1.0)
            # Deploy: v1 goes away mid-activity, v2 takes over the queue.
            v1.send_signal(signal.SIGTERM)
            v1.wait(timeout=15)
            v2 = _spawn(address, "t-upgrade", str(script_path), journal, "fde_kit.durable.workflow")
            try:
                out = await asyncio.wait_for(h.result(), timeout=120)
                hist = await history_summary(h)
                history = [ev async for ev in h.fetch_history_events()]
            finally:
                v2.send_signal(signal.SIGTERM)
                v2.wait(timeout=15)
            return out, hist, history
        finally:
            if v1.poll() is None:
                v1.kill()

    out, hist, history = loop.run_until_complete(go())

    assert out.output == "Both reports reviewed."
    # The v2 worker completed it, and says so...
    assert out.code_version == CODE_VERSION == 2
    # ...but every step stayed on the v1 path, including the ones executed after the swap.
    assert [s.code_version for s in out.steps] == [1, 1, 1, 2]  # last = no-tool step, stamped
    for step in out.steps[:3]:
        for r in step.tool_results:
            assert "summarized_from_chars" not in r
    assert len(out.steps[2].tool_results[0]["content"]) > 1500  # Q2 report NOT summarized

    # The activity sequence is exactly what v1 would have produced: no summarize_result.
    names = [a["activity"] for a in hist["activities"]]
    assert names == [
        "llm_call",
        "tool_execute",
        "llm_call",
        "tool_execute",
        "llm_call",
        "tool_execute",
        "llm_call",
    ]
    # No workflow task ever failed (that is what a non-determinism error looks like in history).
    assert hist["counts"].get("EVENT_TYPE_WORKFLOW_TASK_FAILED", 0) == 0
    # And no patch marker was recorded: this run never entered the new branch.
    markers = [ev for ev in history if ev.HasField("marker_recorded_event_attributes")]
    assert markers == []


def test_v1_histories_replay_through_v2_code(env, tmp_path: Path):
    """The pre-deploy check: every open v1 run must replay cleanly on the new code."""
    e, loop = env

    async def go():
        # Produce a complete v1 history in-process.
        stop = asyncio.Event()
        w = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-replay",
                provider=MockProvider(SCRIPT["script"], SCRIPT["utility_reply"]),
                registry=__import__("fde_kit.durable.demo_tools", fromlist=["registry"]).registry,
                journal_dir=str(tmp_path),
                stop=stop,
                workflow_cls=AgentWorkflowV1,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("review", client=e.client, task_queue="t-replay")
        out = await h.result()
        assert out.code_version == 1 and all(s.code_version == 1 for s in out.steps)
        hist = await h.fetch_history()
        stop.set()
        await w

        # Replay that history through the live (v2) workflow class.
        await replayer([AgentWorkflow]).replay_workflow(hist)  # raises on non-determinism
        return hist

    hist = loop.run_until_complete(go())
    assert len(hist.events) > 10


def test_v2_from_start_takes_new_branch_and_records_marker(env, tmp_path: Path):
    e, loop = env

    async def go():
        stop = asyncio.Event()
        w = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-v2",
                provider=MockProvider(SCRIPT["script"], SCRIPT["utility_reply"]),
                registry=__import__("fde_kit.durable.demo_tools", fromlist=["registry"]).registry,
                journal_dir=str(tmp_path),
                stop=stop,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("review", client=e.client, task_queue="t-v2")
        out = await h.result()
        hist = await history_summary(h)
        history = [ev async for ev in h.fetch_history_events()]
        stop.set()
        await w
        return out, hist, history

    out, hist, history = loop.run_until_complete(go())
    assert all(s.code_version == 2 for s in out.steps)
    assert [a["activity"] for a in hist["activities"]].count("summarize_result") == 2
    assert out.steps[0].tool_results[0]["content"] == "SUMMARY: all metrics nominal."
    markers = [ev for ev in history if ev.HasField("marker_recorded_event_attributes")]
    assert (
        len(markers) == 1
        and markers[0].marker_recorded_event_attributes.marker_name == "core_patch"
    )
