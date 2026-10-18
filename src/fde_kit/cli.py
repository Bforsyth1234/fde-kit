"""`fde` command line.

fde run "What is 17 * 23?"          # one prompt through the agent, prints the trace summary
fde eval                              # run the golden set, exit 1 if below threshold
fde tools                             # list registered tools and their schemas
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .builtin_tools import registry
from .config import get_settings
from .tracing import init_tracing

console = Console()


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    init_tracing(settings)
    agent = Agent(registry, settings=settings)
    result = agent.run(args.prompt)

    console.rule("[bold]answer")
    console.print(result.output)
    console.rule("[bold]run")
    table = Table(show_header=True)
    table.add_column("turn")
    table.add_column("tool calls")
    table.add_column("errors")
    table.add_column("ms", justify="right")
    for s in result.steps:
        names = ", ".join(tc["name"] for tc in s.tool_calls) or "-"
        errs = sum(1 for r in s.tool_results if r["is_error"])
        table.add_row(str(s.turn), names, str(errs), f"{s.latency_ms:.0f}")
    console.print(table)
    console.print(
        f"stopped: {result.stopped_reason}  tokens: {result.usage.input_tokens} in / "
        f"{result.usage.output_tokens} out  total: {result.total_ms:.0f} ms"
    )
    return 0


def cmd_tools(_: argparse.Namespace) -> int:
    for schema in registry.api_schemas():
        console.print(json.dumps(schema, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    # `evals/` lives in the repo root, not inside the installed package. Make it importable
    # whether we're run from the repo root or anywhere else with an editable install.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (Path.cwd(), repo_root):
        if (candidate / "evals" / "run_evals.py").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from evals.run_evals import main as eval_main
    except ModuleNotFoundError:
        console.print(
            "[red]could not find evals/run_evals.py[/]. Run from the repo root, or use "
            "`python -m evals.run_evals`."
        )
        return 2

    argv = []
    if args.provider:
        argv += ["--provider", args.provider]
    if args.threshold is not None:
        argv += ["--threshold", str(args.threshold)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.ids:
        argv += ["--ids", args.ids]
    if args.json:
        argv += ["--json", args.json]
    return eval_main(argv)


def cmd_worker(args: argparse.Namespace) -> int:
    try:
        from .durable.worker import main as worker_main
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2
    argv = ["--address", args.address, "--task-queue", args.task_queue]
    if args.provider:
        argv += ["--provider", args.provider]
    return worker_main(argv)


def cmd_durable_run(args: argparse.Namespace) -> int:
    try:
        import asyncio

        from .durable.client import connect, history_summary, start_run
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2

    async def go() -> int:
        client = await connect(args.address)
        handle = await start_run(args.prompt, client=client, task_queue=args.task_queue)
        console.print(
            f"workflow id: {handle.id}  (UI: http://localhost:8233/namespaces/default/workflows/{handle.id})"
        )
        out = await handle.result()
        console.rule("[bold]answer")
        console.print(out.output)
        hist = await history_summary(handle)
        table = Table(title="activities")
        table.add_column("#", justify="right")
        table.add_column("activity")
        table.add_column("attempts", justify="right")
        for i, a in enumerate(hist["activities"], 1):
            table.add_row(str(i), a["activity"], str(a["attempts"]))
        console.print(table)
        console.print(
            f"stopped: {out.stopped_reason}  "
            f"tokens: {out.input_tokens} in / {out.output_tokens} out"
        )
        return 0

    return asyncio.run(go())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fde", description="fde-kit command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker", help="run a Temporal worker for durable agent runs")
    w.add_argument("--address", default="localhost:7233")
    w.add_argument("--task-queue", default="fde-agent")
    w.add_argument("--provider", choices=["anthropic", "mock"])
    w.set_defaults(fn=cmd_worker)

    d = sub.add_parser("durable-run", help="start a durable run and wait for the result")
    d.add_argument("prompt")
    d.add_argument("--address", default="localhost:7233")
    d.add_argument("--task-queue", default="fde-agent")
    d.set_defaults(fn=cmd_durable_run)

    r = sub.add_parser("run", help="run one prompt through the agent")
    r.add_argument("prompt")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("tools", help="list registered tools")
    t.set_defaults(fn=cmd_tools)

    e = sub.add_parser("eval", help="run the golden eval set")
    e.add_argument("--provider", choices=["anthropic", "mock"])
    e.add_argument("--threshold", type=float)
    e.add_argument("--limit", type=int)
    e.add_argument("--ids", help="comma-separated case ids to run")
    e.add_argument("--json", help="write full per-case results to this file")
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
