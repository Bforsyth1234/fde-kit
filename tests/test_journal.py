"""Idempotency journal backends.

The Postgres tests need a reachable database. Set `FDE_TEST_PG_URL` (any Postgres you can
write to; a throwaway `docker compose up postgres` works) or they skip. CI starts a service
container for them.

The last test is the reason the Postgres backend exists: two workers that do not share a disk,
and a crash in the window between "tool finished and was journaled" and "Temporal recorded the
activity as complete". With a file journal on worker A's disk, worker B re-executes the tool.
With Postgres, worker B finds A's record and returns it without running the tool again.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from fde_kit.durable.journal import FileJournal, PostgresJournal, journal_from_url
from fde_kit.durable.models import ToolCallOutput

PG_URL = os.environ.get("FDE_TEST_PG_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")


def test_file_journal_round_trip(tmp_path: Path):
    j = FileJournal(tmp_path)
    assert j.get("k") is None
    j.put("k", ToolCallOutput(content="hello", is_error=False), tool_name="t")
    got = j.get("k")
    assert got == ToolCallOutput(content="hello", is_error=False, from_journal=True)
    assert j.count() == 1


def test_journal_from_url_dispatch(tmp_path: Path):
    assert isinstance(journal_from_url(None), FileJournal)
    assert isinstance(journal_from_url(f"file:{tmp_path}"), FileJournal)
    assert isinstance(journal_from_url(str(tmp_path)), FileJournal)


@needs_pg
def test_postgres_journal_round_trip_and_first_write_wins():
    j = PostgresJournal(PG_URL)
    key = f"test:{uuid.uuid4().hex}"
    try:
        assert j.get(key) is None
        j.put(key, ToolCallOutput(content="first", is_error=False), tool_name="t")
        j.put(key, ToolCallOutput(content="second", is_error=True), tool_name="t")  # no-op
        got = j.get(key)
        assert got == ToolCallOutput(content="first", is_error=False, from_journal=True)
    finally:
        j.close()


@needs_pg
def test_postgres_journal_visible_across_instances():
    """Two separate pools (two workers) see each other's records."""
    a, b = PostgresJournal(PG_URL), PostgresJournal(PG_URL)
    key = f"test:{uuid.uuid4().hex}"
    try:
        a.put(key, ToolCallOutput(content="from A"), tool_name="t")
        assert b.get(key).content == "from A"
    finally:
        a.close()
        b.close()


def test_file_journal_isolates_tenants_on_colliding_key(tmp_path: Path):
    """Two tenants that choose the same workflow_id and tool_use_id must never read each other's
    journaled tool outputs."""
    j = FileJournal(tmp_path)
    key = "shared-wf:1:toolu_mock_0_0"
    j.put(
        key,
        ToolCallOutput(content="Harbor late fee is $75"),
        tool_name="search_documents",
        tenant_id="harbor",
    )
    assert j.get(key, tenant_id="summit", tool_name="search_documents") is None
    assert j.get(key, tenant_id=None, tool_name="search_documents") is None
    assert j.get(key, tenant_id="harbor", tool_name="other_tool") is None
    hit = j.get(key, tenant_id="harbor", tool_name="search_documents")
    assert hit is not None and hit.content == "Harbor late fee is $75"


@needs_pg
def test_postgres_journal_isolates_tenants_on_colliding_key():
    j = PostgresJournal(PG_URL)
    key = f"shared-wf:{uuid.uuid4().hex}:1:toolu_mock_0_0"
    try:
        j.put(
            key,
            ToolCallOutput(content="Harbor late fee is $75"),
            tool_name="search_documents",
            tenant_id="harbor",
        )
        assert j.get(key, tenant_id="summit", tool_name="search_documents") is None
        assert j.get(key, tenant_id=None, tool_name="search_documents") is None
        assert j.get(key, tenant_id="harbor", tool_name="other_tool") is None
        hit = j.get(key, tenant_id="harbor", tool_name="search_documents")
        assert hit is not None and hit.content == "Harbor late fee is $75"
    finally:
        j.close()


# ------------------------------------------------------------- the crash-window scenario

temporalio = pytest.importorskip("temporalio")


def test_worker_bound_to_one_tenant_rejects_another_tenants_activity(tmp_path: Path):
    """A worker whose registry is bound to one tenant must not run another tenant's step, even
    if a misrouted task queue hands it one. (Unbound workers take the tenant from the input.)"""
    from fde_kit import MockProvider, ToolRegistry
    from fde_kit.durable.activities import AgentActivities

    reg = ToolRegistry(tenant_id="summit")
    acts = AgentActivities(MockProvider(), reg, journal=FileJournal(tmp_path))
    assert acts._effective_tenant("summit") == "summit"
    assert acts._effective_tenant(None) == "summit"
    with pytest.raises(PermissionError):
        acts._effective_tenant("harbor")
    unbound = AgentActivities(MockProvider(), ToolRegistry(), journal=FileJournal(tmp_path))
    assert unbound._effective_tenant("harbor") == "harbor"


from temporalio.testing import WorkflowEnvironment  # noqa: E402

from fde_kit.durable.client import start_run  # noqa: E402

COUNTER_TOOLS = "tests.fixtures.counting_tools:registry"


def _spawn(address, queue, script_path, journal_args, counter_file) -> subprocess.Popen:
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
        COUNTER_TOOLS,
        *journal_args,
    ]
    env = {
        **os.environ,
        "FDE_OTEL_ENABLED": "false",
        "PYTHONPATH": os.getcwd(),
        "FDE_COUNTER_FILE": counter_file,
        "FDE_TEST_HANG_AFTER_JOURNAL": "8",
    }
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_crash_window(env, tmp_path: Path, journal_args_a, journal_args_b):
    """Start a run whose tool (a) does a side effect, (b) records it, (c) is journaled, (d) then
    hangs so the worker can be killed before Temporal records completion. Return the number
    of real side-effect executions and the final output."""
    e, loop = env
    address = e.client.service_client.config.target_host
    script = [
        {"tool_calls": [{"name": "charge_card", "input": {"amount": 42}}]},
        "Charged.",
    ]
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))
    counter = tmp_path / "charges.txt"
    queue = f"t-crash-window-{uuid.uuid4().hex[:6]}"

    async def go():
        a = _spawn(address, queue, str(script_path), journal_args_a, str(counter))
        try:
            await asyncio.sleep(1.5)
            h = await start_run("charge", client=e.client, task_queue=queue)
            # Wait for the tool to have executed AND journaled, then be hanging.
            deadline = time.time() + 30
            while time.time() < deadline:
                if counter.exists() and counter.read_text().count("CHARGED") >= 1:
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("tool never executed")
            await asyncio.sleep(1.0)  # ensure the journal write happened
            a.send_signal(signal.SIGKILL)
            a.wait(timeout=10)
            b = _spawn(address, queue, str(script_path), journal_args_b, str(counter))
            try:
                out = await asyncio.wait_for(h.result(), timeout=90)
            finally:
                b.send_signal(signal.SIGTERM)
                b.wait(timeout=10)
            return out
        finally:
            if a.poll() is None:
                a.kill()

    out = loop.run_until_complete(go())
    executions = counter.read_text().count("CHARGED")
    return executions, out, None


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


@pytest.mark.durable
def test_file_journal_on_separate_disks_double_executes(env, tmp_path: Path):
    """Documents the failure the Postgres backend fixes. Two workers, two journal dirs."""
    executions, out, _ = _run_crash_window(
        env,
        tmp_path,
        ["--journal-dir", str(tmp_path / "disk-A")],
        ["--journal-dir", str(tmp_path / "disk-B")],
    )
    assert out.output == "Charged."
    assert executions == 2, "expected the tool to run twice with unshared file journals"
    assert out.steps[0].tool_results[0]["from_journal"] is False


@needs_pg
@pytest.mark.durable
def test_postgres_journal_prevents_double_execution_across_workers(env, tmp_path: Path):
    """The headline. Same crash window, shared Postgres journal: the card is charged once."""
    executions, out, _ = _run_crash_window(
        env,
        tmp_path,
        ["--journal-url", PG_URL],
        ["--journal-url", PG_URL],
    )
    assert out.output == "Charged."
    assert executions == 1, f"card charged {executions} times"
    assert out.steps[0].tool_results[0]["from_journal"] is True
    assert "charged $42" in out.steps[0].tool_results[0]["content"]
