"""`fde` command line.

fde run "What is 17 * 23?"          # one prompt through the agent, prints the trace summary
fde eval                              # run the golden set, exit 1 if below threshold
fde tools                             # list registered tools and their schemas
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
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .builtin_tools import registry
from .config import get_settings
from .tracing import init_tracing

console = Console()


def cmd_run(args: argparse.Namespace) -> int:
    _check_run_flags(args)
    settings = get_settings()
    init_tracing(settings)
    if args.replay or args.fork:
        return _cmd_run_from_record(args, settings)

    agent = Agent(registry, settings=settings)
    if args.record:
        result = agent.run(args.prompt, record=Path(args.record))
    else:
        if settings.record_runs:
            _attach_recorder(agent, args.prompt, settings)
        result = agent.run(args.prompt)
    _print_run(result)
    if result.record_path:
        console.print(f"run_id {result.run_id} saved to {result.record_path}")
    return 0


def _print_run(result) -> None:
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
    _print_sources(result)


# --- run/replay flags (2-D)
def _check_run_flags(args: argparse.Namespace) -> None:
    """Argparse-level errors (exit 2) for flag combinations the parser cannot express."""
    parser: argparse.ArgumentParser = args.parser
    if args.record and (args.replay or args.fork):
        parser.error("--record cannot be combined with --replay or --fork")
    if args.replay and args.fork:
        parser.error("--replay and --fork are mutually exclusive")
    if args.at is not None and not args.fork:
        parser.error("--at only makes sense with --fork")
    if args.live_llm and not args.replay:
        parser.error("--live-llm only makes sense with --replay")
    if args.fork and args.at is None:
        parser.error("--fork requires --at N")
    if args.fork and not args.prompt:
        parser.error("--fork requires an instruction (the positional prompt)")
    if not args.replay and not args.fork and not args.prompt:
        parser.error("a prompt is required (or --replay PATH / --fork PATH)")


def _attach_recorder(agent: Agent, prompt: str, settings, session=None) -> Path:
    """Open a RunRecorder at `<runs_dir>/<run_id>.jsonl` on `agent` for its next run.

    The run_id comes from the header, so the file is named before the run starts and no rename
    is needed afterwards.
    """
    from .record import RunRecorder, default_runs_dir

    header = agent.build_header(prompt, None, session)
    path = default_runs_dir(settings) / f"{header['run_id']}.jsonl"
    agent.recorder = RunRecorder(path, header)
    return path


def _live_provider(settings):
    from .llm import make_provider

    return make_provider(
        settings.provider,
        settings.model,
        api_key=settings.anthropic_api_key,
        workspace_id=settings.anthropic_workspace_id,
    )


def _divergence_line(result) -> str:
    from .record import ReplayDiverged

    div = result.divergence or {}
    return str(ReplayDiverged(result.diverged_at or 0, div.get("expected"), div.get("got")))


def _print_divergence(result) -> int:
    from rich.markup import escape

    console.print(f"[red]{escape(_divergence_line(result))}[/]")
    return 3


def _first_difference(recorded: str, replayed: str) -> str:
    """`identical`, or `differs` plus the first line that is not the same."""
    if recorded == replayed:
        return "identical"
    a, b = recorded.splitlines(), replayed.splitlines()
    for n, (x, y) in enumerate(zip(a, b, strict=False), 1):
        if x != y:
            return f"differs at line {n}:\n  recorded: {x}\n  replayed: {y}"
    n = min(len(a), len(b)) + 1
    extra = (a if len(a) > len(b) else b)[n - 1] if max(len(a), len(b)) >= n else ""
    which = "recorded" if len(a) > len(b) else "replayed"
    return f"differs at line {n}:\n  only in {which}: {extra}"


def _cmd_run_from_record(args: argparse.Namespace, settings) -> int:
    from rich.markup import escape

    from .record import (
        RunRecord,
        fork_session,
        live_llm_session,
        recorded_prompt,
        replay_session,
    )

    path = Path(args.replay or args.fork)
    try:
        record = RunRecord.load(path)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 2
    prompt = recorded_prompt(record)
    pinned = record.header.get("prompt_text")

    if args.fork:
        live = _live_provider(settings)
        session = fork_session(record, registry, live, at=args.at, instruction=args.prompt)
        agent = Agent(registry, provider=live, settings=settings, system_prompt=pinned)
        _attach_recorder(agent, prompt, settings, session)
        result = agent.run(prompt, session=session)
        _print_run(result)
        if result.stopped_reason == "replay_diverged":
            return _print_divergence(result)
        console.print(f"forked {record.run_id} at turn {args.at} -> {result.record_path}")
        return 0

    if args.live_llm:
        session = live_llm_session(record, registry, _live_provider(settings))
    else:
        session = replay_session(record, registry)
    agent = Agent(registry, provider=session.provider, settings=settings, system_prompt=pinned)
    result = agent.run(prompt, session=session)
    if result.stopped_reason == "replay_diverged":
        return _print_divergence(result)
    recorded_out = (record.result or {}).get("output", "")
    console.rule("[bold]recorded")
    console.print(recorded_out, markup=False)
    console.rule("[bold]replayed" + (" (live llm)" if args.live_llm else ""))
    console.print(result.output, markup=False)
    console.rule("[bold]run")
    console.print(
        f"stopped: {result.stopped_reason}  turns: {result.turns}  "
        f"total: {result.total_ms:.0f} ms"
    )
    _print_sources(result)
    console.print(_first_difference(recorded_out, result.output), markup=False)
    return 0


# --- end run/replay flags (2-D)


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


# --- durable export (1-B)
def cmd_durable_export(args: argparse.Namespace) -> int:
    try:
        import asyncio
        from pathlib import Path

        from .durable.client import connect, handle_for
        from .durable.export import export_run
    except ModuleNotFoundError:
        console.print("[red]temporalio not installed[/]: pip install 'fde-kit[durable]'")
        return 2

    async def go() -> int:
        client = await connect(args.address)
        h = handle_for(client, args.workflow_id)
        path = await export_run(h, out=Path(args.out) if args.out else None)
        console.print(f"exported {h.id} to {path}")
        return 0

    return asyncio.run(go())


# --- end durable export (1-B)


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
            agent = Agent(reg, settings=settings)
            _attach_recorder(agent, args.question, settings)  # always records (design doc)
            result = agent.run(args.question)
            console.rule(f"[bold]{args.tenant}")
            console.print(result.output)
            _print_sources(result)
            console.print(f"recorded {result.record_path}")
            return 0
        # --- rag facts (2-D)
        if args.rag_cmd == "query":
            with store.session(args.tenant) as s:
                try:
                    res = s.query(args.sql, max_rows=args.max_rows)
                except rag.SqlRejected as exc:
                    console.print(f"[red]Query rejected[/]: {exc.reason}")
                    return 1
            conflict_col = res.columns.index("conflict_group_id") if (
                "conflict_group_id" in res.columns
            ) else None
            table = Table(title=f"{args.tenant}: {args.sql}")
            for col in res.columns:
                table.add_column(col)
            if conflict_col is not None:
                table.add_column("conflict")
            for row in res.rows:
                cells = ["" if v is None else str(v) for v in row]
                if conflict_col is not None:
                    cells.append("[red]true[/]" if row[conflict_col] is not None else "")
                table.add_row(*cells)
            console.print(table)
            note = f" (truncated at {args.max_rows}; add a WHERE or LIMIT)" if res.truncated else ""
            console.print(f"{len(res.rows)} row(s) in {res.elapsed_ms:.1f} ms{note}")
            return 0
        if args.rag_cmd == "fetch":
            with store.session(args.tenant) as s:
                try:
                    d = s.fetch(args.doc_id, char_start=args.start, char_end=args.end)
                except KeyError:
                    console.print("[red]No document with that id.[/]")
                    return 1
            console.print(
                f"{d.title or d.source} ({d.source}) v{d.version} "
                f"chars {d.char_start}-{d.char_end} of {d.total_chars}",
                markup=False,
                highlight=False,
                style="bold",
            )
            console.rule()
            console.print(d.text, markup=False, highlight=False)
            if d.next is not None:
                console.print(f"\nnext: {d.next}")
            return 0
        if args.rag_cmd == "dictionary":
            with store.session(args.tenant) as s:
                entries = s.dictionary()
            table = Table(title=f"{args.tenant}: field dictionary")
            for col in ("field", "unit", "definition"):
                table.add_column(col)
            for e in entries:
                table.add_row(e.field, e.unit, e.definition)
            console.print(table)
            return 0
        # --- end rag facts (2-D)
    return 2


# --- adapters (PR 4)
EDGAR_TENANTS = {"pharma-a": "Pharma A", "pharma-b": "Pharma B"}
"""The EDGAR demo tenants (`fde rag seed-edgar`); their filings come from the manifest."""

UA_MISSING = (
    "FDE_EDGAR_USER_AGENT is not set. SEC requires a contact in the User-Agent for sec.gov "
    "requests: export FDE_EDGAR_USER_AGENT='<app name> <contact>', or pass --offline to use "
    "the committed fixtures."
)


def edgar_fixtures_dir() -> Path:
    """The committed offline cache (evals/fixtures/edgar/cache): the current directory's copy
    when run from a checkout, else the one next to the source tree."""
    rel = Path("evals") / "fixtures" / "edgar" / "cache"
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (base / rel).is_dir():
            return base / rel
    return Path(__file__).resolve().parents[2] / rel


def edgar_client(*, offline: bool, settings=None):
    """Offline: the committed fixtures, no contact needed. Online: the live cache
    (settings.edgar_cache_dir) and the contact from settings; ValueError when it is empty."""
    from .rag.adapters import EdgarClient

    settings = settings or get_settings()
    if offline:
        return EdgarClient("", cache_dir=edgar_fixtures_dir(), offline=True)
    ua = (settings.edgar_user_agent or "").strip()
    if not ua:
        raise ValueError(UA_MISSING)
    return EdgarClient(ua, cache_dir=Path(settings.edgar_cache_dir))


def edgar_filings(tenant: str, tickers: list[str] | None = None):
    """The tenant's pinned filings, optionally limited to some tickers (ValueError on a ticker
    the tenant's manifest does not have)."""
    from .rag.adapters import filings_for

    filings = filings_for(tenant)
    if not tickers:
        return filings
    wanted = {t.upper() for t in tickers}
    unknown = wanted - {f.ticker.upper() for f in filings}
    if unknown:
        raise ValueError(f"tenant {tenant!r} has no pinned filings for {sorted(unknown)}")
    return [f for f in filings if f.ticker.upper() in wanted]


def ingest_edgar(store, tenant: str, client, tickers: list[str] | None = None):
    """Ingest a tenant's pinned filings through `adapters.ingest` in one tenant session."""
    from .rag.adapters import EdgarAdapter, ingest

    adapter = EdgarAdapter(client, edgar_filings(tenant, tickers))
    with store.session(tenant) as s:
        return ingest(s, adapter)


def seed_edgar(store, admin_url: str, *, offline: bool = True, client=None, migrate: bool = True):
    """Create the EDGAR tenants and ingest each one's manifest filings. Idempotent: a re-run
    writes no documents and no facts. Returns {tenant: IngestReport}."""
    from .rag import create_tenant
    from .rag import migrate as rag_migrate

    if migrate:
        rag_migrate(admin_url)
    client = client or edgar_client(offline=offline)
    reports = {}
    for tid, name in EDGAR_TENANTS.items():
        create_tenant(admin_url, tid, name)
        reports[tid] = ingest_edgar(store, tid, client)
    return reports


def _print_ingest_reports(reports: dict) -> None:
    table = Table(title="ingest")
    table.add_column("tenant", no_wrap=True)
    for col in ("entities", "docs", "changed", "facts", "conflicts"):
        table.add_column(col, justify="right")
    table.add_column("job")
    for tid, r in reports.items():
        table.add_row(
            tid, str(r.entities), str(r.documents), str(r.documents_changed), str(r.facts),
            str(r.conflict_groups), r.job_id[:8] or "-",
        )
    console.print(table)


def _adapter_store(args: argparse.Namespace):
    from . import rag

    _, app = _rag_urls(args)
    embedder = rag.make_embedder(getattr(args, "embedder", None) or get_settings().rag_embedder)
    return rag.TenantStore(app, embedder)


def cmd_rag_ingest(args: argparse.Namespace) -> int:
    from rich.markup import escape

    from .rag.adapters import EdgarCacheMiss, LocalFilesAdapter, ingest

    if args.adapter == "edgar":
        if args.dir or args.entity:
            console.print("[red]--dir/--entity are for --adapter local_files[/]")
            return 2
        try:
            client = edgar_client(offline=args.offline)
            edgar_filings(args.tenant, args.ticker)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            return 2
        with _adapter_store(args) as store:
            try:
                report = ingest_edgar(store, args.tenant, client, args.ticker)
            except EdgarCacheMiss as exc:
                console.print(f"[red]{escape(str(exc))}[/]")
                return 1
        _print_ingest_reports({args.tenant: report})
        return 0

    if not args.dir:
        console.print("[red]--adapter local_files needs --dir PATH[/]")
        return 2
    if args.ticker or args.offline:
        console.print("[red]--ticker/--offline are for --adapter edgar[/]")
        return 2
    root = Path(args.dir)
    if not root.is_dir():
        console.print(f"[red]not a directory[/]: {escape(str(root))}")
        return 2
    adapter = LocalFilesAdapter(root, entity=args.entity)
    with _adapter_store(args) as store, store.session(args.tenant) as s:
        report = ingest(s, adapter)
    _print_ingest_reports({args.tenant: report})
    for rel in adapter.skipped:
        console.print(f"[yellow]skipped[/] {escape(rel)}")
    return 0


def cmd_rag_seed_edgar(args: argparse.Namespace) -> int:
    from rich.markup import escape

    from . import rag

    try:
        client = edgar_client(offline=args.offline)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 2
    admin, _ = _rag_urls(args)
    rag.migrate(admin)  # before the store opens: its posture check needs the app role
    with _adapter_store(args) as store:
        reports = seed_edgar(store, admin, client=client, migrate=False)
    _print_ingest_reports(reports)
    return 0


def cmd_rag_edgar_drift(args: argparse.Namespace) -> int:
    from rich.markup import escape

    from .rag.adapters import check_drift

    try:
        client = edgar_client(offline=False)  # live cache dir, never the fixtures
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 2
    problems = check_drift(client)
    for msg in problems:
        console.print(f"[red]DRIFT[/] {escape(msg)}")
    if not problems:
        console.print("[green]ok[/]: every pinned filing matches the manifest")
    return 1 if problems else 0


def _add_adapter_parsers(rsub) -> None:
    ri = rsub.add_parser("ingest", help="load a source into one tenant through an adapter")
    ri.add_argument("--tenant", required=True)
    ri.add_argument("--adapter", required=True, choices=["edgar", "local_files"])
    ri.add_argument("--ticker", action="append", metavar="TICKER", help="edgar: only these")
    ri.add_argument("--offline", action="store_true", help="edgar: the committed fixtures")
    ri.add_argument("--dir", help="local_files: the directory to ingest")
    ri.add_argument("--entity", help="local_files: an entity to register for the files")
    ri.add_argument("--url")
    ri.add_argument("--embedder")
    ri.set_defaults(fn=cmd_rag_ingest)
    se = rsub.add_parser("seed-edgar", help="create pharma-a / pharma-b and ingest the manifest")
    se.add_argument("--offline", action="store_true", help="the committed fixtures, no network")
    se.add_argument("--admin-url")
    se.add_argument("--url")
    se.add_argument("--embedder")
    se.set_defaults(fn=cmd_rag_seed_edgar)
    dr = rsub.add_parser("edgar-drift", help="re-fetch the pinned filings; exit 1 on drift")
    dr.set_defaults(fn=cmd_rag_edgar_drift)


# --- end adapters (PR 4)


def _print_sources(result) -> None:
    """Evidence the run saw, marking what the answer actually cited."""
    if not result.sources:
        return
    cited = {s.source_id for s in result.cited}
    console.rule("[bold]sources")
    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("cited")
    table.add_column("source")
    table.add_column("title")
    table.add_column("hash")
    for n, s in enumerate(result.sources, 1):
        mark = "[green]yes[/]" if s.source_id in cited else "-"
        table.add_row(str(n), mark, s.uri, s.title, (s.content_hash or "")[:12])
    console.print(table)
    if result.fabricated_citations:
        console.print(
            f"[red]fabricated citations[/]: {result.fabricated_citations} "
            "(numbers in the answer that no tool returned)"
        )


# --- runs (1-C)


class RunFileError(ValueError):
    """A run file that is not the §2 format (missing/invalid header, bad JSON)."""


def _load_run_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one `runs/<run_id>.jsonl` file via `RunRecord.load`; RunFileError on a bad file."""
    from .record import RunRecord

    try:
        rec = RunRecord.load(path)
    except ValueError as exc:
        raise RunFileError(str(exc)) from exc
    return rec.header, rec.rows


def _runs_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "dir", None):
        return Path(args.dir)
    return Path(get_settings().runs_dir)


def _scan_runs(dir_: Path) -> tuple[list[tuple[Path, dict, list[dict]]], list[tuple[Path, str]]]:
    """Every *.jsonl in `dir_`: (path, header, rows) when readable, else (path, why)."""
    good: list[tuple[Path, dict, list[dict]]] = []
    bad: list[tuple[Path, str]] = []
    if not dir_.is_dir():
        return good, bad
    for path in sorted(dir_.glob("*.jsonl")):
        try:
            header, rows = _load_run_file(path)
        except (RunFileError, OSError, UnicodeDecodeError) as exc:
            bad.append((path, str(exc)))
            continue
        good.append((path, header, rows))
    return good, bad


def _result_row(rows: list[dict]) -> dict | None:
    for row in reversed(rows):
        if row.get("kind") == "result":
            return row
    return None


def _parent_label(header: dict) -> str:
    parent = header.get("parent_run_id")
    if not parent:
        return "-"
    return f"{parent}@{header.get('fork_at')}"


def _prompt_label(header: dict) -> str:
    sha = str(header.get("prompt_sha256") or "")
    return f"{header.get('prompt_name') or '-'}@{sha[:12]}"


def _find_run(
    runs: list[tuple[Path, dict, list[dict]]], run_id: str
) -> tuple[Path, dict, list[dict]] | None:
    for entry in runs:
        path, header, _ = entry
        if header.get("run_id") == run_id or path.stem == run_id:
            return entry
    return None


def cmd_runs(args: argparse.Namespace) -> int:
    dir_ = _runs_dir(args)
    runs, bad = _scan_runs(dir_)

    if args.runs_cmd == "list":
        if not runs and not bad:
            console.print(f"no runs in {dir_}")
            return 0
        runs.sort(key=lambda e: str(e[1].get("created_at") or ""), reverse=True)
        table = Table(title=f"runs in {dir_}", show_header=True)
        for col in (
            "run_id", "created_at", "runtime", "tenant", "model", "prompt", "turns",
            "stopped", "parent",
        ):
            table.add_column(col)
        for _, header, rows in runs:
            result = _result_row(rows)
            turns = str(result.get("turns", "-")) if result else "-"
            stopped = str(result.get("stopped_reason") or "-") if result else "incomplete"
            table.add_row(
                str(header.get("run_id")),
                str(header.get("created_at") or "-"),
                str(header.get("runtime") or "-"),
                str(header.get("tenant_id") or "-"),
                str(header.get("model") or "-"),
                _prompt_label(header),
                turns,
                stopped,
                _parent_label(header),
            )
        for path, _why in bad:
            table.add_row(path.stem, "-", "-", "-", "-", "-", "-", "[red]unreadable[/]", "-")
        console.print(table)
        return 0

    if args.runs_cmd == "tree":
        target = _find_run(runs, args.run_id)
        if target is None:
            console.print(f"[red]unknown run[/] {args.run_id} (looked in {dir_})")
            return 1
        by_id = {h["run_id"]: (p, h, r) for p, h, r in runs}
        children: dict[str, list[str]] = {}
        for _, h, _ in runs:
            parent = h.get("parent_run_id")
            if parent:
                children.setdefault(parent, []).append(h["run_id"])

        # Walk up to the root (stop on a missing parent or a cycle).
        chain: list[str] = []
        cur: str | None = target[1]["run_id"]
        seen: set[str] = set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = by_id[cur][1].get("parent_run_id")
        chain.reverse()
        missing_parent = cur if cur and cur not in by_id else None

        def label(rid: str) -> str:
            h = by_id[rid][1]
            at = h.get("fork_at")
            mark = "  [bold]<-- this run[/]" if rid == target[1]["run_id"] else ""
            fork = f" (fork of {h.get('parent_run_id')} at turn {at})" if at is not None else ""
            return f"{rid}  {h.get('created_at') or ''}{fork}{mark}"

        depth = 0
        if missing_parent:
            console.print(f"{missing_parent}  [dim](parent not in {dir_})[/]")
            depth = 1
        for rid in chain:
            console.print("  " * depth + label(rid), soft_wrap=True)
            depth += 1

        def walk(rid: str, d: int) -> None:
            for child in sorted(
                children.get(rid, []), key=lambda c: str(by_id[c][1].get("created_at") or "")
            ):
                if child in seen:
                    continue
                seen.add(child)
                console.print("  " * d + label(child), soft_wrap=True)
                walk(child, d + 1)

        walk(target[1]["run_id"], depth)
        return 0

    if args.runs_cmd == "show":
        target = _find_run(runs, args.run_id)
        if target is None:
            console.print(f"[red]unknown run[/] {args.run_id} (looked in {dir_})")
            return 1
        path, header, rows = target
        console.rule(f"[bold]{header['run_id']}  ({path})")
        for key, value in header.items():
            if key == "prompt_text":
                continue
            console.print(f"{key}: {json.dumps(value) if not isinstance(value, str) else value}")
        console.rule("[bold]rows")
        for row in rows:
            console.print(_describe_row(row))
        return 0
    return 2


def _describe_row(row: dict) -> str:
    kind = row.get("kind", "?")
    turn = row.get("turn", "-")
    head = f"{turn!s:>4}  {kind:<6} "
    if kind == "llm":
        names = ", ".join(tc.get("name", "?") for tc in row.get("tool_calls") or []) or "-"
        return head + f"stop_reason={row.get('stop_reason')}  tools: {names}"
    if kind == "tool":
        n = len(row.get("sources") or [])
        return head + f"{row.get('name')}  is_error={row.get('is_error')}  sources={n}"
    if kind == "user":
        text = " ".join(str(row.get("text", "")).split())[:60]
        return head + f"{row.get('actor', 'user')}: {text}"
    if kind == "result":
        out = " ".join(str(row.get("output", "")).split())[:80]
        n = len(row.get("cited") or [])
        return f"   -  result stopped={row.get('stopped_reason')}  cited={n}  {out}"
    return head + json.dumps({k: v for k, v in row.items() if k not in ("kind", "turn")})[:80]


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

    # --- durable export (1-B)
    de = sub.add_parser(
        "durable-export", help="write a finished durable run's history to runs/<run_id>.jsonl"
    )
    de.add_argument("workflow_id")
    de.add_argument("--out", help="write the record to this path instead of runs/")
    de.add_argument("--address", default="localhost:7233")
    de.set_defaults(fn=cmd_durable_export)
    # --- end durable export (1-B)

    r = sub.add_parser("run", help="run one prompt through the agent")
    r.add_argument("prompt", nargs="?", help="the prompt; the fork instruction with --fork")
    # --- run/replay flags (2-D)
    r.add_argument("--record", metavar="PATH", help="record this run to PATH")
    r.add_argument("--replay", metavar="PATH", help="replay a recorded run exactly")
    r.add_argument(
        "--live-llm", action="store_true", help="with --replay: tools from the record, model live"
    )
    r.add_argument("--fork", metavar="PATH", help="fork a recorded run after turn --at")
    r.add_argument("--at", type=int, metavar="N", help="with --fork: replay turns 1..N exactly")
    r.set_defaults(fn=cmd_run, parser=r)
    # --- end run/replay flags (2-D)

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
    # --- rag facts (2-D)
    rq = rsub.add_parser("query", help="run one read-only SELECT over a tenant's fact tables")
    rq.add_argument("sql")
    rq.add_argument("--tenant", required=True)
    rq.add_argument("--max-rows", type=int, default=200)
    rq.add_argument("--url")
    rq.add_argument("--embedder")
    rf = rsub.add_parser("fetch", help="print a slice of one of a tenant's documents")
    rf.add_argument("doc_id")
    rf.add_argument("--tenant", required=True)
    rf.add_argument("--start", type=int, default=0)
    rf.add_argument("--end", type=int, default=6000)
    rf.add_argument("--url")
    rf.add_argument("--embedder")
    rd = rsub.add_parser("dictionary", help="print a tenant's field dictionary")
    rd.add_argument("--tenant", required=True)
    rd.add_argument("--url")
    rd.add_argument("--embedder")
    # --- end rag facts (2-D)
    # --- adapters (PR 4)
    _add_adapter_parsers(rsub)  # these subcommands set their own fn, overriding cmd_rag
    # --- end adapters (PR 4)
    rg.set_defaults(fn=cmd_rag)

    # --- runs (1-C)
    ru = sub.add_parser("runs", help="inspect recorded runs (runs/*.jsonl)")
    rusub = ru.add_subparsers(dest="runs_cmd", required=True)
    rl = rusub.add_parser("list", help="table of every recorded run, newest first")
    rl.add_argument("--dir", help="runs directory (default: $FDE_RUNS_DIR or ./runs)")
    rt = rusub.add_parser("tree", help="lineage of one run: parents above, forks below")
    rt.add_argument("run_id")
    rt.add_argument("--dir")
    rsh = rusub.add_parser("show", help="header and one line per recorded row")
    rsh.add_argument("run_id")
    rsh.add_argument("--dir")
    ru.set_defaults(fn=cmd_runs)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
