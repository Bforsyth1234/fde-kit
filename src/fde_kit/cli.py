"""`fde` command line.

fde run "What is 17 * 23?"          # one prompt through the agent, prints the trace summary
fde eval                              # run the golden set, exit 1 if below threshold
fde tools                             # list registered tools and their schemas
fde keygen                            # payload key: Temporal history + journal become ciphertext
fde rag migrate                       # create RAG roles, tables and RLS policies
fde rag seed-demo                     # three demo tenants on a shared lease template
fde rag search --tenant summit "late fee?"
fde rag ask --tenant summit "How much is the late fee?"
fde rag posture                       # would this connection let RLS be bypassed?
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


def cmd_keygen(_: argparse.Namespace) -> int:
    """Print a fresh payload key. Put it in .env as FDE_PAYLOAD_KEY on every worker and client."""
    try:
        from .durable.crypto import PayloadCipher
    except ModuleNotFoundError:
        console.print("[red]cryptography not installed[/]: pip install 'fde-kit[durable]'")
        return 2
    print(f"FDE_PAYLOAD_KEY={PayloadCipher.generate().export_current()}")
    return 0


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


def _rag_urls(args: argparse.Namespace) -> tuple[str, str]:
    from .rag import app_url_from_admin

    settings = get_settings()
    admin = getattr(args, "admin_url", None) or settings.rag_admin_url
    app = getattr(args, "url", None) or settings.rag_url or app_url_from_admin(admin)
    return admin, app


def cmd_rag(args: argparse.Namespace) -> int:
    try:
        from . import rag
        from .rag.demo_corpus import TENANTS, seed
    except ModuleNotFoundError as exc:
        console.print(f"[red]{exc.name} not installed[/]: pip install 'fde-kit[rag]'")
        return 2
    settings = get_settings()
    admin, app = _rag_urls(args)

    if args.rag_cmd == "migrate":
        rag.migrate(admin)
        console.print(f"migrated. app role: {rag.APP_ROLE}")
        return 0
    if args.rag_cmd == "create-tenant":
        rag.create_tenant(admin, args.tenant_id, args.name or "")
        console.print(f"tenant {args.tenant_id} ready")
        return 0
    if args.rag_cmd == "posture":
        import psycopg

        with psycopg.connect(app) as conn:
            problems = rag.check_posture(conn)
        for p in problems:
            console.print(f"[red]FAIL[/] {p}")
        if not problems:
            console.print("[green]ok[/]: RLS applies to this connection on every RAG table")
        return 1 if problems else 0

    embedder = rag.make_embedder(getattr(args, "embedder", None) or settings.rag_embedder)
    with rag.TenantStore(app, embedder) as store:
        if args.rag_cmd == "seed-demo":
            rag.migrate(admin)
            for tid, t in TENANTS.items():
                rag.create_tenant(admin, tid, t["name"])
            console.print(f"seeded chunks per tenant: {seed(store)}")
            return 0
        if args.rag_cmd == "search":
            with store.session(args.tenant) as s:
                hits = s.search(args.query, k=args.k, mode=args.mode)
            table = Table(title=f"{args.tenant}: {args.query}")
            for col in ("#", "score", "vec", "kw", "source", "text"):
                table.add_column(col)
            for i, h in enumerate(hits, 1):
                snippet = " ".join(h.content.split("\n\n", 1)[-1].split())[:90]
                table.add_row(str(i), f"{h.score:.3f}", str(h.vector_rank or "-"),
                              str(h.keyword_rank or "-"), h.source, snippet)
            console.print(table)
            return 0
        if args.rag_cmd == "ask":
            init_tracing(settings)
            reg = rag.tenant_search_registry(store, args.tenant)
            result = Agent(reg, settings=settings).run(args.question)
            console.rule(f"[bold]{args.tenant}")
            console.print(result.output)
            return 0
    return 2


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

    kg = sub.add_parser("keygen", help="generate an FDE_PAYLOAD_KEY for durable-run encryption")
    kg.set_defaults(fn=cmd_keygen)

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

    rg = sub.add_parser("rag", help="multi-tenant retrieval")
    rsub = rg.add_subparsers(dest="rag_cmd", required=True)
    for name, help_ in (
        ("migrate", "create RAG roles, tables and policies (admin URL)"),
        ("seed-demo", "create the three demo tenants and ingest the lease corpus"),
        ("posture", "check that RLS cannot be bypassed on the app connection"),
    ):
        sp = rsub.add_parser(name, help=help_)
        sp.add_argument("--admin-url")
        sp.add_argument("--url")
        sp.add_argument("--embedder")
    ct = rsub.add_parser("create-tenant", help="register a tenant (admin URL)")
    ct.add_argument("tenant_id")
    ct.add_argument("--name")
    ct.add_argument("--admin-url")
    rs = rsub.add_parser("search", help="search as one tenant")
    rs.add_argument("query")
    rs.add_argument("--tenant", required=True)
    rs.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    rs.add_argument("-k", type=int, default=5)
    rs.add_argument("--url")
    rs.add_argument("--embedder")
    ra = rsub.add_parser("ask", help="run the agent with search bound to one tenant")
    ra.add_argument("question")
    ra.add_argument("--tenant", required=True)
    ra.add_argument("--url")
    ra.add_argument("--embedder")
    rg.set_defaults(fn=cmd_rag)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
