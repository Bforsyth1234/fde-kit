"""`fde` command line.

fde run "What is 17 * 23?"          # one prompt through the agent, prints the trace summary
fde eval                              # run the golden set, exit 1 if below threshold
fde tools                             # list registered tools and their schemas
fde rag migrate                       # create RAG roles, tables and RLS policies
fde rag seed-demo                     # three demo tenants on a shared lease template
fde rag search --tenant summit "late fee?"
fde rag ask --tenant summit "How much is the late fee?"
fde rag posture                       # would this connection let RLS be bypassed?
fde wrangle dry-run SPEC FILE...      # a mapping spec over messy CSV/XLSX, nothing written
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

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
    if len(a) == len(b):
        # Every line matches, so the strings differ only in line breaks: a trailing newline,
        # or \r\n vs \n (splitlines() treats both as the same boundary).
        return f"differs only in line breaks:\n  recorded: {recorded!r}\n  replayed: {replayed!r}"
    n = min(len(a), len(b)) + 1
    longer, which = (a, "recorded") if len(a) > len(b) else (b, "replayed")
    return f"differs at line {n}:\n  only in {which}: {longer[n - 1]}"


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


# --- wrangle (PR 5)
WRANGLE_MISSING = "[red]{name} not installed[/]: pip install 'fde-kit[wrangle]'"
DEFAULT_MAX_REJECT_RATE = 0.02
"""§8's default: apply refuses when rejects / (facts + rejects) is over this."""


def _wrangle_files(files: list[str]) -> list[Path] | None:
    """The FILE arguments as paths, or None (after printing which) when one isn't a file."""
    from rich.markup import escape

    paths = [Path(f) for f in files]
    missing = [p for p in paths if not p.is_file()]
    for p in missing:
        console.print(f"[red]not a file[/]: {escape(str(p))}")
    return None if missing else paths


def _wrangle_check(spec_path: str):
    """Load the spec with its dictionary and crosswalks, compile its ops and check each
    crosswalk for norm_key collisions. Returns (loaded or None, problems)."""
    from .wrangle import check_crosswalk, load_project_spec, run

    try:
        loaded = load_project_spec(spec_path)
        run(loaded, [])  # compiles every op pipeline without reading a file
    except ValueError as exc:  # SpecError, pydantic.ValidationError
        return None, [str(exc)]
    problems = [p for cw in loaded.crosswalks.values() for p in check_crosswalk(cw)]
    return loaded, problems


def _approval_line(loaded) -> str:
    from rich.markup import escape

    approval = loaded.spec.approval
    if approval is None:
        return "[yellow]not approved[/]"

    who = f"by {escape(approval.by)} at {escape(str(approval.at))}"
    if loaded.approved:
        return f"[green]approved[/] {who}"
    return (
        f"[red]stale approval[/] {who}: stamped at bundle {approval.bundle_sha256[:12]}, "
        f"the bundle is now {loaded.bundle_sha12}"
    )


def cmd_wrangle_profile(args: argparse.Namespace) -> int:
    from rich.markup import escape

    try:
        from .wrangle import profile_file
    except ModuleNotFoundError as exc:
        console.print(WRANGLE_MISSING.format(name=exc.name))
        return 2
    if _wrangle_files([args.file]) is None:
        return 2
    try:
        prof = profile_file(args.file, args.sheet, encoding=args.encoding)
    except ValueError as exc:  # unknown sheet or encoding, a sheet name given for a CSV, ...
        console.print(f"[red]{escape(str(exc))}[/]")
        return 1
    if args.json:
        sys.stdout.write(prof.model_dump_json(indent=2) + "\n")
        return 0

    console.rule(f"[bold]{escape(prof.file)} — {escape(prof.sheet)}")
    found = "" if prof.header_found else "  [yellow](guessed: no row looked like a header)[/]"
    console.print(
        f"{prof.n_rows} rows x {prof.n_cols} cols  header rows {prof.header_rows}{found}  "
        f"data rows {prof.data.start}-{prof.data.end} ({prof.data.rows} non-empty)"
    )
    console.print(f"header_fingerprint {prof.header_fingerprint}")
    if prof.encoding is not None:
        console.print(f"encoding {prof.encoding}")
    for tr in prof.title_rows:
        text = "  ".join(f"{c.ref}={c.value}" for c in tr.cells)
        console.print(f"title row {tr.row}: {escape(text)}")

    table = Table(title="columns")
    for col in ("col", "label", "type", "nulls", "distinct", "format", "scale", "samples"):
        table.add_column(col)
    for c in prof.columns:
        if c.number_format is not None:
            nf = c.number_format
            fmt = f"number {nf.thousands!r}/{nf.decimal!r}" + (" ?" if nf.ambiguous else "")
            extras = [
                k
                for k in ("parens_negative", "trailing_minus", "currency", "percent")
                if getattr(nf, k)
            ]
            fmt += "".join(f" +{k}" for k in extras)
        elif c.date_formats:
            fmt = c.date_formats[0] + (" (d/m ambiguous)" if c.day_month_ambiguous else "")
        else:
            fmt = "-"
        scale = f"x{c.scale_hint.scale}" if c.scale_hint else "-"
        samples = ", ".join(s.value for s in c.samples[:3])
        table.add_row(
            c.letter,
            escape(c.label),
            c.type,
            f"{c.null_rate:.0%}",
            str(c.distinct),
            escape(fmt),
            scale,
            escape(samples),
        )
    console.print(table)

    for s in prof.skip_candidates:
        sums = f" (sums: {', '.join(s.sum_columns)})" if s.sum_columns else ""
        console.print(
            f"skip candidate row {s.row}: {'+'.join(s.reasons)} {escape(s.first_value)!r}"
            f"{escape(sums)}"
        )
    for u in prof.unpivot_candidates:
        console.print(
            f"unpivot candidate {escape(u.from_)} .. {escape(u.to)} "
            f"({len(u.columns)} cols {u.columns[0]}-{u.columns[-1]}, "
            f"{escape(u.format or u.pattern or '')}, grain {u.grain})"
        )
    for h in prof.scale_hints:
        console.print(f"scale hint x{h.scale} from {h.ref or h.source}: {escape(h.text)}")
    for key in prof.keys:
        console.print(f"key candidate: {escape(' + '.join(key))}")
    for row in prof.notes_rows:
        text = "  ".join(f"{c.ref}={c.value}" for c in row.cells)
        console.print(f"notes row {row.row}: {escape(text)}")
    for w in prof.warnings:
        console.print(f"[yellow]warning[/] {escape(w)}")
    return 0


def cmd_wrangle_check(args: argparse.Namespace) -> int:
    from rich.markup import escape

    try:
        loaded, problems = _wrangle_check(args.spec)
    except ModuleNotFoundError as exc:
        console.print(WRANGLE_MISSING.format(name=exc.name))
        return 2
    for p in problems:
        console.print(f"[red]FAIL[/] {escape(p)}")
    if loaded is None:
        return 1
    console.print(f"spec {loaded.spec.spec}  bundle_sha256 {loaded.bundle_sha256}")
    for path, sha in sorted(loaded.files.items()):
        console.print(f"  {escape(path)}  {sha[:12]}")
    console.print(_approval_line(loaded))
    if not problems:
        console.print("[green]ok[/]: the spec, dictionary and crosswalks validate")
    return 1 if problems else 0


def cmd_wrangle_approve(args: argparse.Namespace) -> int:
    from rich.markup import escape

    try:
        from .wrangle import approve

        loaded, problems = _wrangle_check(args.spec)
    except ModuleNotFoundError as exc:
        console.print(WRANGLE_MISSING.format(name=exc.name))
        return 2
    if not args.by.strip():
        console.print("[red]--by must not be empty[/]")
        return 2
    for p in problems:
        console.print(f"[red]FAIL[/] {escape(p)}")
    if problems:
        console.print("[red]refused[/]: `fde wrangle check` fails, so the spec was not approved")
        return 1
    stamped = approve(args.spec, by=args.by)
    console.print(
        f"approved {stamped.spec.spec} at bundle {stamped.bundle_sha256} by {escape(args.by)}"
    )
    return 0


WRANGLE_SHOWN_CONFLICTS = 20


def _print_wrangle_result(result) -> None:
    from rich.markup import escape

    rep = result.report
    approved = "approved" if result.approved else "not approved"
    console.rule(f"[bold]{escape(rep['spec'])}  bundle {result.loaded.bundle_sha12}  ({approved})")
    table = Table(title="files")
    table.add_column("file", overflow="fold")  # the whole path, wrapped, never "…"
    table.add_column("sheet")
    for col in ("rows", "skipped", "empty", "facts"):
        table.add_column(col, justify="right")
    for f, fr in zip(rep["files"], result.files, strict=True):
        table.add_row(
            escape(f["path"]),
            escape(f["sheet"]),
            str(f["rows_read"]),
            str(sum(f["rows_skipped"].values())),
            str(f["empty_skipped"]),
            str(len(fr.facts)),
        )
    console.print(table)
    for f in rep["files"]:
        sec = f.get("section") or {}
        if sec:
            bounds = "  ".join(f"{k} {v}" for k, v in sec.items())
            console.print(f"section {escape(f['sheet'])}: {bounds}")
        nulls = f.get("null_skipped") or {}
        if nulls:
            shown = "  ".join(f"{escape(k)!r} {v}" for k, v in nulls.items())
            where = f" ({escape(f['sheet'])})" if len(rep["files"]) > 1 else ""
            console.print(f"placeholders skipped as empty{where}: {shown}")
    rj = rep["rejects"]
    console.print(
        f"facts {rep['facts']}  rejects {rj['total']} (rate {rj['rate']:.4f})  "
        f"duplicates dropped {rep['duplicates_dropped']}"
    )
    for reason, n in rj["by_reason"].items():
        console.print(f"  reject {n} x {escape(reason)}")
    ents = "  ".join(f"{k} {v}" for k, v in rep["entities"].items())
    console.print(f"entities by strategy: {ents}")
    for p in result.proposed:
        console.print(
            f"  [yellow]fuzzy[/] {escape(p.alias)!r} -> {escape(p.canonical)!r} "
            f"({p.score:g} vs {p.second:g}); see crosswalk.proposed.csv"
        )
    conflicts = rep["conflicts_within_load"]
    console.print(f"conflicts within load: {len(conflicts)}")
    for c in conflicts[:WRANGLE_SHOWN_CONFLICTS]:
        console.print(f"  {escape(' / '.join(c['key']))}: {escape(', '.join(c['cells']))}")
    if len(conflicts) > WRANGLE_SHOWN_CONFLICTS:
        more = len(conflicts) - WRANGLE_SHOWN_CONFLICTS
        console.print(f"  ... and {more} more (all of them are in report.json with --out)")
    similar = rep["similar_entities"]
    console.print(f"possible duplicate entities: {len(similar)}")
    for d in similar[:WRANGLE_SHOWN_CONFLICTS]:
        console.print(
            f"  [yellow]{d['score']:g}[/] {escape(d['part'])}: {escape(d['a'])!r} ~ "
            f"{escape(d['b'])!r} ({escape(d['reason'])}; {escape(', '.join(d['cells']))})"
        )
    if len(similar) > WRANGLE_SHOWN_CONFLICTS:
        more = len(similar) - WRANGLE_SHOWN_CONFLICTS
        console.print(
            f"  ... and {more} more (all of them are in similar.csv with --out; "
            "add reviewed pairs to resolve.not_same)"
        )


def _print_gate(reasons: list[str], max_reject_rate: float, *, now: bool) -> None:
    from rich.markup import escape

    if reasons:
        verdict = "apply now would refuse" if now else "refused, nothing written"
        console.print(f"[red]{verdict}[/] (max reject rate {max_reject_rate}):")
        for r in reasons:
            console.print(f"  - {escape(r)}")
    else:
        console.print(f"[green]apply now would load[/] (max reject rate {max_reject_rate})")


def _print_outputs(paths: dict) -> None:
    from rich.markup import escape

    for name, path in paths.items():
        console.print(f"wrote {name}: {escape(str(path))}", soft_wrap=True)


def _wrangle_run(args: argparse.Namespace):
    """`run(spec, files)` for dry-run/apply: (result, None), or (None, exit code) after printing
    why. Bad FILE arguments are 2; a SpecMismatch or a spec that doesn't load is 1."""
    from rich.markup import escape

    from .wrangle import SpecMismatch, run

    files = _wrangle_files(args.files)
    if files is None:
        return None, 2
    try:
        return run(args.spec, files, force=args.force), None
    except SpecMismatch as exc:
        console.print(f"[red]spec mismatch[/]: {escape(str(exc))}")
    except ValueError as exc:  # SpecError, pydantic.ValidationError
        console.print(f"[red]spec error[/]: {escape(str(exc))}")
    return None, 1


def cmd_wrangle_dry_run(args: argparse.Namespace) -> int:
    try:
        from .wrangle import gate, write_outputs

        result, code = _wrangle_run(args)
    except ModuleNotFoundError as exc:
        console.print(WRANGLE_MISSING.format(name=exc.name))
        return 2
    if result is None:
        return code
    _print_wrangle_result(result)
    _print_gate(gate(result), DEFAULT_MAX_REJECT_RATE, now=True)
    if args.out:
        _print_outputs(write_outputs(result, args.out))
    return 0


def cmd_wrangle_apply(args: argparse.Namespace) -> int:
    """Gate first, store second: the approval and reject-rate checks run on the dry-run result
    before any database connection opens, so a refused apply writes nothing (§8). The load
    itself is `apply_to_tenant` inside one `store.session(tenant)` transaction."""
    from rich.markup import escape

    try:
        from .wrangle import gate, write_outputs

        result, code = _wrangle_run(args)
    except ModuleNotFoundError as exc:
        console.print(WRANGLE_MISSING.format(name=exc.name))
        return 2
    if result is None:
        return code
    rate = args.max_reject_rate
    reasons = gate(result, draft=args.draft, max_reject_rate=rate)
    _print_wrangle_result(result)
    if reasons:
        _print_gate(reasons, rate, now=False)
        if args.out:
            _print_outputs(write_outputs(result, args.out))
        return 1

    try:
        from .wrangle.adapter import apply_to_tenant, job_kind_for

        store_cm = _adapter_store(args)
    except ModuleNotFoundError as exc:
        console.print(f"[red]{exc.name} not installed[/]: pip install 'fde-kit[rag]'")
        return 2
    with store_cm as store, store.session(args.tenant) as s:
        applied, report = apply_to_tenant(
            s,
            result.loaded,
            [Path(f) for f in args.files],
            draft=args.draft,
            max_reject_rate=rate,
            force=args.force,
        )
    if args.out:
        _print_outputs(write_outputs(applied, args.out))
    if report is None:  # pragma: no cover - the same gates already passed above
        _print_gate(gate(applied, draft=args.draft, max_reject_rate=rate), rate, now=False)
        return 1
    table = Table(title=f"apply: {escape(args.tenant)}")
    for col in ("entities", "docs", "changed", "facts", "replaced", "conflicts"):
        table.add_column(col, justify="right")
    table.add_column("job")
    table.add_row(
        str(report.entities),
        str(report.documents),
        str(report.documents_changed),
        str(report.facts),
        str(report.facts_replaced),
        str(report.conflict_groups),
        report.job_id[:8] or "-",
    )
    console.print(table)
    console.print(f"job kind {escape(job_kind_for(applied.loaded))}")
    return 0


def _add_wrangle_parsers(sub) -> None:
    wg = sub.add_parser("wrangle", help="messy CSV/XLSX tables -> cited facts, via a mapping spec")
    wsub = wg.add_subparsers(dest="wrangle_cmd", required=True)
    pr = wsub.add_parser("profile", help="what is in a sheet: header, types, totals, layout")
    pr.add_argument("file")
    pr.add_argument("--sheet", help="XLSX sheet (default: the first)")
    pr.add_argument(
        "--encoding",
        help="CSV encoding (default: try utf-8-sig, then cp1252, then latin-1, and say which)",
    )
    pr.add_argument("--json", action="store_true", help="print the profile model as JSON")
    pr.set_defaults(fn=cmd_wrangle_profile)
    ck = wsub.add_parser("check", help="validate a spec and its crosswalks; print the bundle hash")
    ck.add_argument("spec")
    ck.set_defaults(fn=cmd_wrangle_check)
    dr = wsub.add_parser("dry-run", help="apply a spec to files without writing to a database")
    dr.add_argument("spec")
    dr.add_argument("files", nargs="+", metavar="FILE")
    dr.add_argument("--out", help="write report.json, rejects.jsonl, previews here")
    dr.add_argument("--force", action="store_true", help="skip the header fingerprint check")
    dr.set_defaults(fn=cmd_wrangle_dry_run)
    ap = wsub.add_parser("approve", help="stamp the spec's approval at its current bundle hash")
    ap.add_argument("spec")
    ap.add_argument("--by", required=True)
    ap.set_defaults(fn=cmd_wrangle_approve)
    al = wsub.add_parser("apply", help="load a spec's facts into one tenant (gated, §8)")
    al.add_argument("spec")
    al.add_argument("files", nargs="+", metavar="FILE")
    al.add_argument("--tenant", required=True)
    al.add_argument("--max-reject-rate", type=float, default=DEFAULT_MAX_REJECT_RATE)
    al.add_argument("--draft", action="store_true", help="load an unapproved spec, tagged draft")
    al.add_argument("--force", action="store_true", help="skip the header fingerprint check")
    al.add_argument("--out", help="write report.json, rejects.jsonl, previews here")
    al.add_argument("--url")
    al.add_argument("--embedder")
    al.set_defaults(fn=cmd_wrangle_apply)


# --- end wrangle (PR 5)


def _print_sources(result) -> None:
    """Evidence the run saw, marking what the answer actually cited.

    A compact table (number, cited, title, hash) and then one line per source with its FULL
    location. The table shrinks to the terminal width and would cut a long SEC URL to
    "https://www.sec.gov/Archives/edgar/data/78…"; the lines below it are never truncated or
    wrapped, so each can be copied whole, and http(s) links are emitted as terminal hyperlinks
    where the terminal supports them.
    """
    if not result.sources:
        return
    cited = {s.source_id for s in result.cited}
    console.rule("[bold]sources")
    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("cited")
    table.add_column("title")
    table.add_column("hash")
    for n, s in enumerate(result.sources, 1):
        mark = "[green]yes[/]" if s.source_id in cited else "-"
        table.add_row(str(n), mark, s.title or s.uri, (s.content_hash or "")[:12])
    console.print(table)
    width = len(str(len(result.sources)))
    for n, s in enumerate(result.sources, 1):
        line = Text(f"{n:>{width}}  ")
        style = f"link {s.uri}" if s.uri.startswith(("http://", "https://")) else ""
        line.append(s.uri, style=style)
        console.print(line, soft_wrap=True, no_wrap=True, crop=False, overflow="ignore")
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
            console.print(f"no runs in {dir_}", soft_wrap=True)
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
            console.print(f"[red]unknown run[/] {args.run_id} (looked in {dir_})", soft_wrap=True)
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

    if args.runs_cmd == "drift":
        return _cmd_runs_drift(args, runs, dir_)

    if args.runs_cmd == "show":
        target = _find_run(runs, args.run_id)
        if target is None:
            console.print(f"[red]unknown run[/] {args.run_id} (looked in {dir_})", soft_wrap=True)
            return 1
        path, header, rows = target
        console.rule(f"[bold]{header['run_id']}  ({path})")
        for key, value in header.items():
            if key == "prompt_text":
                continue
            console.print(f"{key}: {json.dumps(value) if not isinstance(value, str) else value}")
        console.rule("[bold]rows")
        full = getattr(args, "full", False)
        for row in rows:
            console.print(_describe_row(row), markup=False)
            if full:
                for line in _row_detail(row):
                    console.print(line, markup=False, soft_wrap=True)
        return 0
    return 2


_DRIFT_STYLE = {
    "unchanged": "green",
    "superseded": "yellow",
    "gone": "red",
    "value_changed": "red",
    "new_conflict": "yellow",
    "unchecked": "dim",
}


def _cmd_runs_drift(args: argparse.Namespace, runs: list, dir_: Path) -> int:
    """`fde runs drift RUN_ID`: is the evidence the run saw still what the database holds?
    Exit 0 no drift, 1 drift, 2 usage error. Logic in fde_kit.rag.drift."""
    from rich.markup import escape

    target = _find_run(runs, args.run_id)
    if target is None:
        console.print(f"[red]unknown run[/] {escape(args.run_id)} (looked in {dir_})")
        return 2
    try:
        from . import rag
        from .rag.drift import TenantError, check_drift, resolve_tenant
        from .record import RunRecord
    except ModuleNotFoundError as exc:
        console.print(f"[red]{exc.name} not installed[/]: pip install 'fde-kit[rag]'")
        return 2
    rec = RunRecord.load(target[0])
    try:
        resolve_tenant(rec, args.tenant)
    except TenantError as exc:
        console.print(f"[red]refused[/]: {escape(str(exc))}", soft_wrap=True)
        return 2
    _, app = _rag_urls(args)
    # Drift never embeds anything (no search runs), so the cheap embedder is always enough.
    with rag.TenantStore(app, rag.HashEmbedder()) as store:
        report = check_drift(rec, store, args.tenant)

    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
        return 1 if report.drifted else 0
    table = Table(title=f"drift: {report.run_id} (tenant {report.tenant_id})", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("tool")
    table.add_column("status")
    table.add_column("detail")
    for i in report.items:
        style = _DRIFT_STYLE.get(i.status, "")
        status = i.status + (f" (+{', '.join(i.flags)})" if i.flags else "")
        table.add_row(
            str(i.number),
            escape(i.title or i.uri),
            i.tool,
            f"[{style}]{status}[/]",
            escape(i.detail),
        )
    console.print(table)
    console.print(report.summary(), markup=False)
    return 1 if report.drifted else 0


FULL_CHARS = 1200
"""`fde runs show --full` prints at most this many characters of any one text field."""


def _clip(text: str, limit: int = FULL_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} more chars]"


def _row_detail(row: dict) -> list[str]:
    """What `--full` adds under a row: the model's text and tool inputs, a tool's input and
    output, the final answer. Indented so the one-line summary above stays scannable."""
    pad = "          "
    kind = row.get("kind")
    out: list[str] = []
    if kind == "llm":
        if row.get("text"):
            out.append(pad + "text: " + _clip(row["text"]))
        for tc in row.get("tool_calls") or []:
            out.append(pad + f"call {tc.get('name')}: " + _clip(json.dumps(tc.get("input"))))
    elif kind == "tool":
        inp = row.get("input") or {}
        if set(inp) == {"sql"}:
            out.append(pad + "sql: " + _clip(inp["sql"]))
        else:
            out.append(pad + "input: " + _clip(json.dumps(inp)))
        content = _clip(row.get("content", "")).replace("\n", "\n" + pad + "  ")
        out.append(pad + ("error: " if row.get("is_error") else "output: ") + content)
    elif kind == "result":
        out.append(pad + "output: " + _clip(row.get("output", "")).replace("\n", "\n" + pad))
    return out


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
    rsh.add_argument(
        "--full", action="store_true", help="also print tool inputs/outputs (SQL, results)"
    )
    rdr = rusub.add_parser(
        "drift", help="is the evidence a run saw still in the database? exit 1 on drift"
    )
    rdr.add_argument("run_id")
    rdr.add_argument("--dir")
    rdr.add_argument("--tenant", help="default: the run header's tenant_id (must match it)")
    rdr.add_argument("--json", action="store_true", help="machine-readable results")
    rdr.add_argument("--url", help="app database URL (default: $FDE_RAG_URL or from admin URL)")
    ru.set_defaults(fn=cmd_runs)

    # --- wrangle (PR 5)
    _add_wrangle_parsers(sub)
    # --- end wrangle (PR 5)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
