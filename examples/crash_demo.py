"""Kill the worker mid-run. Watch the run finish anyway.

    docker compose up -d                     # Temporal dev server on :7233, UI on :8233
    python examples/crash_demo.py            # mock provider, no API key needed
    python examples/crash_demo.py --live     # real model

What it does:
  1. starts a worker subprocess
  2. starts an agent run whose first tool call blocks for several seconds
  3. waits until the LLM has decided on that tool and the tool is executing
  4. SIGKILLs the worker (no graceful shutdown, no cleanup)
  5. shows the workflow is still RUNNING with nobody serving it
  6. starts a new worker
  7. prints the final answer and a per-activity table from the event history:
     the first LLM call ran once, the interrupted tool ran twice, everything after ran once

Open http://localhost:8233 while it runs to watch the same thing in the Temporal UI.
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
QUEUE = "fde-crash-demo"
MOCK_SCRIPT = [
    {"tool_calls": [{"name": "wait_seconds", "input": {"seconds": 8}}]},
    {"tool_calls": [{"name": "calculator", "input": {"expression": "17 * 23"}}]},
    "Done. The slow step completed and 17 * 23 = 391.",
]
PROMPT = (
    "First call wait_seconds with seconds=8. After it returns, use the calculator to compute "
    "17 * 23. Then reply with one sentence stating that the slow step completed and the product."
)


def spawn_worker(address: str, live: bool, script_path: str, journal_dir: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "fde_kit.durable.worker",
        "--address",
        address,
        "--task-queue",
        QUEUE,
        "--tools",
        "fde_kit.durable.demo_tools:registry",
        "--journal-dir",
        journal_dir,
    ]
    if not live:
        cmd += ["--provider", "mock", "--mock-script", script_path]
    env = {**os.environ, "FDE_OTEL_ENABLED": os.environ.get("FDE_OTEL_ENABLED", "false")}
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def print_activities(title: str, summary: dict) -> None:
    t = Table(title=title)
    t.add_column("#", justify="right")
    t.add_column("activity")
    t.add_column("attempts", justify="right")
    t.add_column("completed", justify="center")
    for i, a in enumerate(summary["activities"], 1):
        # Temporal writes ActivityTaskStarted to history only when the attempt ends, so an
        # in-flight attempt is not yet countable.
        attempts = str(a["attempts"]) if a["attempts"] else "[yellow]in flight[/]"
        t.add_row(
            str(i),
            a["activity"],
            attempts,
            "[green]yes[/]" if a["completed"] else "[yellow]no[/]",
        )
    console.print(t)


async def main(address: str, live: bool) -> int:
    client = await connect(address)
    tmp = Path(tempfile.mkdtemp(prefix="fde-crash-"))
    script_path = tmp / "script.json"
    script_path.write_text(json.dumps(MOCK_SCRIPT))
    journal_dir = str(tmp / "journal")

    console.rule("[bold]1. start worker A")
    w1 = spawn_worker(address, live, str(script_path), journal_dir)
    console.print(f"worker A pid={w1.pid}")
    await asyncio.sleep(1.5)

    console.rule("[bold]2. start agent run")
    handle = await start_run(PROMPT, client=client, task_queue=QUEUE)
    console.print(f"workflow id = {handle.id}")

    console.rule("[bold]3. wait for the slow tool to be in flight")
    t0 = time.time()
    while time.time() - t0 < 60:
        st = await status(handle)
        console.print(f"  turn={st.turn} phase={st.phase} current_tool={st.current_tool}", end="\r")
        if st.phase == "tools" and st.current_tool == "wait_seconds":
            break
        await asyncio.sleep(0.25)
    else:
        console.print("[red]never reached the tool phase; is the worker healthy?[/]")
        w1.kill()
        return 1
    await asyncio.sleep(1.0)
    console.print()
    print_activities("history before the kill", await history_summary(handle))

    console.rule("[bold red]4. SIGKILL worker A")
    w1.send_signal(signal.SIGKILL)
    w1.wait(timeout=10)
    console.print(f"worker A pid={w1.pid} killed (exit {w1.returncode})")

    console.rule("[bold]5. nobody is serving the run")
    desc = await handle.describe()
    console.print(f"workflow status = [bold]{desc.status.name}[/]  (no worker alive)")
    console.print("Temporal will notice the missing heartbeat within ~10 s and reschedule.")
    await asyncio.sleep(3)

    console.rule("[bold]6. start worker B")
    w2 = spawn_worker(address, live, str(script_path), journal_dir)
    console.print(f"worker B pid={w2.pid}")

    console.rule("[bold]7. run completes on worker B")
    t1 = time.time()
    out = await asyncio.wait_for(handle.result(), timeout=180)
    console.print(f"finished {time.time() - t1:.1f} s after worker B started")
    console.print(f"\n[bold]answer:[/] {out.output}")
    console.print(f"tools called: {out.tools_called}  stopped: {out.stopped_reason}")
    print_activities("history after completion", await history_summary(handle))
    console.print(
        "\nRead the table: activity 1 (the turn-1 LLM call) has 1 attempt and was never re-run; "
        "activity 2 (the tool that was executing when the worker died) needed a second attempt; "
        "everything after ran once. The conversation state lived in Temporal, not in the process."
    )

    w2.send_signal(signal.SIGTERM)
    w2.wait(timeout=10)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    p.add_argument("--live", action="store_true", help="use the real model instead of the mock")
    a = p.parse_args()
    raise SystemExit(asyncio.run(main(a.address, a.live)))
