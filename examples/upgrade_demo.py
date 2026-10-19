"""Deploy a new workflow version while an agent run is in flight.

    docker compose up -d
    python examples/upgrade_demo.py

What it does:
  1. starts a v1 worker (tests/fixtures/workflow_v1.py, frozen at the pre-patch commit)
  2. starts run A on it; waits until A is mid-way through a slow tool
  3. stops the v1 worker and starts a v2 worker (the live fde_kit.durable.workflow)
  4. starts run B on v2
  5. prints both runs' per-step code versions and activity sequences

What you should see:
  - run A finishes on the v2 worker with zero failed workflow tasks and ZERO summarize
    activities: it is pinned to the v1 path for its lifetime. Its conversation never changed
    shape because of the deploy.
  - run B, started on v2, summarizes every long result and records the patch marker.

Rolling a workflow forward is three deploys: patched() -> deprecate_patch() -> remove. This is
the first one, the only one with two live code paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fde_kit.durable.client import connect, history_summary, start_run, status

console = Console()
QUEUE = "fde-upgrade-demo"
SCRIPT = {
    "script": [
        {"tool_calls": [{"name": "fetch_report", "input": {"report_id": "Q1"}}]},
        {"tool_calls": [{"name": "wait_seconds", "input": {"seconds": 6}}]},
        {"tool_calls": [{"name": "fetch_report", "input": {"report_id": "Q2"}}]},
        "Both reports reviewed.",
    ],
    "utility_reply": "SUMMARY: 60 lines, all metrics nominal.",
}


def spawn(address: str, script_path: str, journal: str, module: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "fde_kit.durable.worker",
        "--address",
        address,
        "--task-queue",
        QUEUE,
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


async def wait_for_tool(h, name: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = await status(h)
        if st.phase == "tools" and st.current_tool == name:
            return
        await asyncio.sleep(0.25)
    raise RuntimeError(f"never saw {name} in flight")


def show(title: str, out, hist) -> None:
    t = Table(title=title)
    t.add_column("turn", justify="right")
    t.add_column("code ver", justify="center")
    t.add_column("tool")
    t.add_column("result chars", justify="right")
    t.add_column("summarized", justify="center")
    for s in out.steps:
        if not s.tool_results:
            t.add_row(str(s.turn), str(s.code_version), "-", "-", "-")
        for r in s.tool_results:
            t.add_row(
                str(s.turn),
                str(s.code_version),
                r["name"],
                str(len(r["content"])),
                "[green]yes[/]" if "summarized_from_chars" in r else "no",
            )
    console.print(t)
    names = [a["activity"] for a in hist["activities"]]
    failed = hist["counts"].get("EVENT_TYPE_WORKFLOW_TASK_FAILED", 0)
    console.print(f"  activities: {' → '.join(names)}")
    console.print(
        f"  summarize_result calls: {names.count('summarize_result')}   "
        f"failed workflow tasks: {failed}   completed by code_version={out.code_version}"
    )


async def main(address: str) -> int:
    client = await connect(address)
    tmp = Path(tempfile.mkdtemp(prefix="fde-upgrade-"))
    script_path = tmp / "script.json"
    script_path.write_text(json.dumps(SCRIPT))
    journal = str(tmp / "journal")

    console.rule("[bold]1. v1 worker up")
    v1 = spawn(address, str(script_path), journal, "tests.fixtures.workflow_v1")
    await asyncio.sleep(1.5)
    console.print(f"v1 worker pid={v1.pid}  (tests/fixtures/workflow_v1.py)")

    console.rule("[bold]2. run A starts on v1")
    a = await start_run("review Q1 and Q2", client=client, task_queue=QUEUE, workflow_id="run-A")
    await wait_for_tool(a, "wait_seconds")
    await asyncio.sleep(1)
    st = await status(a)
    console.print(
        f"run A: turn {st.turn}, {st.current_tool} in flight, tools so far {st.tools_called}"
    )

    console.rule("[bold yellow]3. deploy: v1 worker down, v2 worker up")
    v1.send_signal(signal.SIGTERM)
    v1.wait(timeout=15)
    v2 = spawn(address, str(script_path), journal, "fde_kit.durable.workflow")
    await asyncio.sleep(1.5)
    console.print(f"v2 worker pid={v2.pid}  (src/fde_kit/durable/workflow.py, CODE_VERSION=2)")

    console.rule("[bold]4. run B starts on v2")
    b = await start_run("review Q1 and Q2", client=client, task_queue=QUEUE, workflow_id="run-B")

    console.rule("[bold]5. both complete on the v2 worker")
    out_a, out_b = await asyncio.gather(
        asyncio.wait_for(a.result(), 180), asyncio.wait_for(b.result(), 180)
    )
    show("run A  (started on v1, finished on v2)", out_a, await history_summary(a))
    show("run B  (started on v2)", out_b, await history_summary(b))
    console.print(
        "\nRun A kept the v1 behaviour to the end: the deploy did not change its conversation. "
        "Run B got the v2 behaviour from its first step. Same worker, same minute, no failures."
    )

    v2.send_signal(signal.SIGTERM)
    v2.wait(timeout=15)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.address)))
