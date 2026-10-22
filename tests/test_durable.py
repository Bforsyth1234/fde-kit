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
from fde_kit.durable.journal import FileJournal  # noqa: E402
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
    journal = FileJournal(tmp_path)
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


# ---------------------------------------------------------------------- approval gates

from fde_kit.durable.client import approve, pending, reject  # noqa: E402
from fde_kit.durable.demo_tools import registry as demo_registry  # noqa: E402

EMAIL_SCRIPT = [
    {
        "tool_calls": [
            {
                "name": "send_email",
                "input": {"to": "ops@example.com", "subject": "Outage", "body": "All clear."},
            }
        ]
    },
    "Done: the email went out.",
]


async def _wait_for_phase(h, phase: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = await status(h)
        if st.phase == phase:
            return st
        await asyncio.sleep(0.1)
    raise AssertionError(f"never reached phase {phase!r}")


def _worker_task(e, queue, script, journal_dir, stop):
    return asyncio.create_task(
        run_worker(
            client=e.client,
            task_queue=queue,
            provider=MockProvider(script),
            registry=demo_registry,
            journal_dir=journal_dir,
            stop=stop,
        )
    )


def test_gated_tool_waits_then_runs_on_approve(env, tmp_path: Path):
    e, loop = env

    async def go():
        stop = asyncio.Event()
        w = _worker_task(e, "t-approve", EMAIL_SCRIPT, str(tmp_path), stop)
        await asyncio.sleep(0.3)
        h = await start_run("notify ops", client=e.client, task_queue="t-approve")
        st = await _wait_for_phase(h, "awaiting_approval")
        p = await pending(h)
        hist_before = await history_summary(h)
        await approve(h, p.tool_use_id, by="reviewer", note="looks right")
        out = await h.result()
        stop.set()
        await w
        return st, p, hist_before, out

    st, p, hist_before, out = loop.run_until_complete(go())
    assert st.pending_approval is not None and st.pending_approval.name == "send_email"
    assert p.input["to"] == "ops@example.com"
    # While parked: one LLM activity done, no tool activity scheduled yet.
    assert [a["activity"] for a in hist_before["activities"]] == ["llm_call"]
    assert out.output == "Done: the email went out."
    assert out.steps[0].approvals == [
        {
            "tool_use_id": p.tool_use_id,
            "name": "send_email",
            "outcome": "approved",
            "by": "reviewer",
            "note": "looks right",
        }
    ]
    assert out.steps[0].tool_results[0]["content"].startswith("email queued to ops@example.com")


def test_rejected_tool_is_not_executed_and_model_is_told(env, tmp_path: Path):
    e, loop = env
    script = [EMAIL_SCRIPT[0], "Understood, I won't send it. Here is the draft instead: ..."]

    async def go():
        stop = asyncio.Event()
        w = _worker_task(e, "t-reject", script, str(tmp_path), stop)
        await asyncio.sleep(0.3)
        h = await start_run("notify ops", client=e.client, task_queue="t-reject")
        await _wait_for_phase(h, "awaiting_approval")
        p = await pending(h)
        await reject(h, p.tool_use_id, by="reviewer", note="wrong recipient")
        out = await h.result()
        hist = await history_summary(h)
        stop.set()
        await w
        return out, hist

    out, hist = loop.run_until_complete(go())
    # No tool_execute activity ever ran.
    assert [a["activity"] for a in hist["activities"]] == ["llm_call", "llm_call"]
    res = out.steps[0].tool_results[0]
    assert res["is_error"] is True and "wrong recipient" in res["content"]
    assert out.steps[0].approvals[0]["outcome"] == "rejected"
    assert out.output.startswith("Understood")


def test_approval_timeout_counts_as_rejection(env, tmp_path: Path):
    e, loop = env
    script = [EMAIL_SCRIPT[0], "Nobody approved in time, so I did not send it."]

    async def go():
        stop = asyncio.Event()
        w = _worker_task(e, "t-timeout", script, str(tmp_path), stop)
        await asyncio.sleep(0.3)
        h = await start_run(
            "notify ops", client=e.client, task_queue="t-timeout", approval_timeout_seconds=2
        )
        out = await asyncio.wait_for(h.result(), timeout=30)
        stop.set()
        await w
        return out

    out = loop.run_until_complete(go())
    a = out.steps[0].approvals[0]
    assert a["outcome"] == "timeout" and a["by"] == "timeout"
    assert "timed out" in out.steps[0].tool_results[0]["content"]
    assert out.output.startswith("Nobody approved")


def test_parked_run_survives_worker_restart_then_approved(env, tmp_path: Path):
    """The headline: no worker alive while the human decides, and the decision still lands."""
    e, loop = env

    async def go():
        stop1 = asyncio.Event()
        w1 = _worker_task(e, "t-park", EMAIL_SCRIPT, str(tmp_path / "j1"), stop1)
        await asyncio.sleep(0.3)
        h = await start_run("notify ops", client=e.client, task_queue="t-park")
        await _wait_for_phase(h, "awaiting_approval")
        p = await pending(h)

        # Take the only worker down. Nothing is serving this task queue now.
        stop1.set()
        await w1
        await asyncio.sleep(1.0)
        desc = await h.describe()
        assert desc.status.name == "RUNNING"

        # Approve with no worker up: the signal is recorded by the server, not by a worker.
        await approve(h, p.tool_use_id, by="reviewer-from-phone")
        await asyncio.sleep(0.5)
        still = await h.describe()
        assert still.status.name == "RUNNING"  # it can't progress until something serves it

        # New worker, fresh process state, fresh provider instance, different journal dir.
        stop2 = asyncio.Event()
        w2 = _worker_task(e, "t-park", EMAIL_SCRIPT, str(tmp_path / "j2"), stop2)
        out = await asyncio.wait_for(h.result(), timeout=60)
        stop2.set()
        await w2
        return out

    out = loop.run_until_complete(go())
    assert out.output == "Done: the email went out."
    assert out.steps[0].approvals[0]["by"] == "reviewer-from-phone"
    assert out.steps[0].tool_results[0]["content"].startswith("email queued")


# ---------------------------------------------------------------------- record export (1-B)

from fde_kit.durable.export import export_run, run_id_for  # noqa: E402


def test_export_then_replay(env, tmp_path: Path):
    """The standard two-tool mock run exports to a section-2 record file.

    TODO(integrator): replay via fde_kit.record.replay_session once record.py lands, and assert
    the replayed output equals the result row's output with stopped_reason == "end_turn".
    """
    e, loop = env

    async def go():
        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_worker(
                client=e.client,
                task_queue="t-export",
                provider=MockProvider(TWO_TOOL_SCRIPT),
                registry=registry,
                journal_dir=tempfile.mkdtemp(),
                stop=stop,
            )
        )
        await asyncio.sleep(0.3)
        h = await start_run("refund doubled?", client=e.client, task_queue="t-export")
        out = await h.result()
        path = await export_run(h, runs_dir=tmp_path / "runs")
        stop.set()
        await worker
        return h.id, out, path

    wf_id, out, path = loop.run_until_complete(go())

    assert path.exists() and path == tmp_path / "runs" / f"{run_id_for(wf_id)}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header, body = rows[0], rows[1:]
    assert header["kind"] == "header"
    assert header["version"] == 1 and header["runtime"] == "durable"
    assert header["run_id"] == run_id_for(wf_id) and header["workflow_id"] == wf_id
    assert header["model"] == "mock" and header["tools"] == ["lookup_policy", "calculator"]
    assert header["prompt_sha256"] and header["prompt_text"]

    kinds = [r["kind"] for r in body]
    assert kinds[0] == "user" and body[0]["text"] == "refund doubled?"
    assert kinds.count("result") == 1 and kinds[-1] == "result"
    assert kinds.count("llm") == out.turns
    assert kinds.count("tool") == len(out.tools_called)
    # Each turn's tool rows follow that turn's llm row, in tool_calls order.
    assert kinds[1:] == ["llm", "tool", "llm", "tool", "llm", "result"]

    llm1 = body[1]
    assert llm1["turn"] == 1 and llm1["stop_reason"] == "tool_use"
    assert llm1["tool_calls"][0]["name"] == "lookup_policy"
    assert set(llm1["usage"]) == {"input_tokens", "output_tokens"}
    tool1 = body[2]
    assert tool1["turn"] == 1 and tool1["tool_use_id"] == llm1["tool_calls"][0]["id"]
    tool2 = body[4]
    assert tool2["turn"] == 2 and tool2["name"] == "calculator"
    assert tool2["input"] == {"expression": "30 * 2"} and tool2["content"] == "60"
    assert tool2["is_error"] is False and tool2["approval"] is None

    result = body[-1]
    assert result["output"] == out.output
    assert result["stopped_reason"] == "end_turn" and result["turns"] == out.turns
    assert result["cited"] == [] and result["fabricated_citations"] == []
