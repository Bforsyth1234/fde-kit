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

        from .durable.client import approve, connect, history_summary, reject, start_run, status
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2

    async def go() -> int:
        client = await connect(args.address)
        handle = await start_run(
            args.prompt,
            client=client,
            task_queue=args.task_queue,
            require_approval_for=args.gate or None,
        )
        console.print(
            f"workflow id: {handle.id}  "
            f"(UI: http://localhost:8233/namespaces/default/workflows/{handle.id})"
        )

        # Wait for the result, but surface approval requests as they appear so a person at
        # this terminal can answer them. `fde approve <id> <tool_use_id>` works from any other
        # terminal too; the run doesn't care where the signal comes from.
        result_task = asyncio.create_task(handle.result())
        answered: set[str] = set()
        while not result_task.done():
            await asyncio.wait({result_task}, timeout=0.5)
            if result_task.done():
                break
            st = await status(handle)
            p = st.pending_approval
            if p and p.tool_use_id not in answered:
                answered.add(p.tool_use_id)
                console.rule("[bold yellow]approval required")
                console.print(f"turn {p.turn}: the agent wants to call [bold]{p.name}[/] with")
                console.print(json.dumps(p.input, indent=2))
                if args.no_prompt:
                    console.print(
                        f"run: fde approve {handle.id} {p.tool_use_id}   "
                        f"or: fde reject {handle.id} {p.tool_use_id} --note '...'"
                    )
                    continue
                ans = await asyncio.to_thread(input, "approve? [y/N/note] ")
                ans = ans.strip()
                if ans.lower() in ("y", "yes"):
                    await approve(handle, p.tool_use_id, by="cli")
                    console.print("[green]approved[/]")
                else:
                    await reject(handle, p.tool_use_id, by="cli", note=ans or "rejected at CLI")
                    console.print("[red]rejected[/]")
        out = result_task.result()
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


def _cmd_decide(args: argparse.Namespace, approved: bool) -> int:
    try:
        import asyncio

        from .durable.client import approve, connect, handle_for, reject
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2

    async def go() -> int:
        client = await connect(args.address)
        h = handle_for(client, args.workflow_id)
        fn = approve if approved else reject
        await fn(h, args.tool_use_id, by=args.by, note=args.note or "")
        console.print(f"{'approved' if approved else 'rejected'} {args.tool_use_id} on {h.id}")
        return 0

    return asyncio.run(go())


def cmd_approve(args: argparse.Namespace) -> int:
    return _cmd_decide(args, approved=True)


def cmd_reject(args: argparse.Namespace) -> int:
    return _cmd_decide(args, approved=False)


def cmd_pending(args: argparse.Namespace) -> int:
    try:
        import asyncio

        from .durable.client import connect, handle_for, status
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2

    async def go() -> int:
        client = await connect(args.address)
        st = await status(handle_for(client, args.workflow_id))
        console.print(f"turn {st.turn}  phase {st.phase}  tools so far {st.tools_called}")
        if st.pending_approval:
            p = st.pending_approval
            console.print(f"[yellow]awaiting approval[/]: {p.name} (tool_use_id {p.tool_use_id})")
            console.print(json.dumps(p.input, indent=2))
        return 0

    return asyncio.run(go())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fde", description="fde-kit command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, help_ in (
        ("approve", cmd_approve, "approve a pending gated tool call on a durable run"),
        ("reject", cmd_reject, "reject a pending gated tool call on a durable run"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("workflow_id")
        sp.add_argument("tool_use_id")
        sp.add_argument("--by", default="cli")
        sp.add_argument("--note")
        sp.add_argument("--address", default="localhost:7233")
        sp.set_defaults(fn=fn)

    pe = sub.add_parser("pending", help="show a durable run's phase and any pending approval")
    pe.add_argument("workflow_id")
    pe.add_argument("--address", default="localhost:7233")
    pe.set_defaults(fn=cmd_pending)

    w = sub.add_parser("worker", help="run a Temporal worker for durable agent runs")
    w.add_argument("--address", default="localhost:7233")
    w.add_argument("--task-queue", default="fde-agent")
    w.add_argument("--provider", choices=["anthropic", "mock"])
    w.set_defaults(fn=cmd_worker)

    d = sub.add_parser("durable-run", help="start a durable run and wait for the result")
    d.add_argument("prompt")
    d.add_argument("--address", default="localhost:7233")
    d.add_argument("--task-queue", default="fde-agent")
    d.add_argument(
        "--gate", action="append", metavar="TOOL", help="require approval for TOOL this run"
    )
    d.add_argument("--no-prompt", action="store_true", help="print instructions instead of asking")
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
