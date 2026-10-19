"""Three property managers, one lease template, one question.

    make up && python examples/rag_demo.py            # bge-small embeddings
    python examples/rag_demo.py --embedder hash       # offline, no model download

Every tenant's late-fee clause is the same sentence with a different number in it. The demo asks
all three the same question and prints what each one gets back. Then it tries the same thing on
a superuser connection, which the store refuses, and runs the search as that superuser anyway
to show what RLS stops applying to. With ANTHROPIC_API_KEY set, it also runs the agent per tenant.
"""

from __future__ import annotations

import argparse
import os
import re

from rich.console import Console
from rich.table import Table

from fde_kit import Agent
from fde_kit.rag import (
    InsecureConnection,
    TenantStore,
    app_url_from_admin,
    create_tenant,
    make_embedder,
    migrate,
    tenant_search_registry,
)
from fde_kit.rag.demo_corpus import TENANTS, seed

QUESTION = "How much extra do I owe if my rent is late?"
console = Console()


def fee_line(text: str) -> str:
    m = re.search(r"late fee of ([^.]+)\.", text)
    return m.group(1) if m else "(no late-fee clause in this chunk)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--admin-url",
        default=os.environ.get("FDE_RAG_ADMIN_URL", "postgresql://fde:fde@localhost:5432/fde"),
    )
    p.add_argument("--embedder", default=os.environ.get("FDE_RAG_EMBEDDER", "bge"))
    a = p.parse_args()

    migrate(a.admin_url)
    for tid, t in TENANTS.items():
        create_tenant(a.admin_url, tid, t["name"])
    embedder = make_embedder(a.embedder)

    with TenantStore(app_url_from_admin(a.admin_url), embedder) as store:
        console.print(f"seeded chunks per tenant: {seed(store)}\n")

        table = Table(title=f'"{QUESTION}"  (app role, RLS enforced)')
        for col in ("tenant", "top source", "late fee in top chunk", "other tenants in top 20"):
            table.add_column(col)
        for tid in TENANTS:
            with store.session(tid) as s:
                hits = s.search(QUESTION, k=20)
            foreign = sum(1 for h in hits if not h.source.startswith(f"{tid}/"))
            table.add_row(tid, hits[0].source, fee_line(hits[0].content), str(foreign))
        console.print(table)

        console.rule("same code, connected as the compose superuser")
        try:
            TenantStore(a.admin_url, embedder)
        except InsecureConnection as exc:
            console.print(f"[yellow]refused[/]: {exc.problems[0]}")
        import psycopg

        with psycopg.connect(a.admin_url) as conn:
            conn.execute("SELECT set_config('app.tenant_id', 'summit', false)")
            rows = conn.execute(
                "SELECT tenant_id, count(*) FROM rag_chunks WHERE tenant_id = ANY(%s) "
                "GROUP BY 1 ORDER BY 1",
                (list(TENANTS),),
            ).fetchall()
        console.print(
            "as that superuser anyway, tenant set to 'summit', policy in place, "
            "SELECT count(*) FROM rag_chunks sees:"
        )
        for tid, n in rows:
            mark = "" if tid == "summit" else "   [red]<- not summit's[/]"
            console.print(f"  {tid:10} {n:5} chunks{mark}")

        if os.environ.get("ANTHROPIC_API_KEY"):
            console.rule("the agent, one registry per tenant")
            for tid in TENANTS:
                reg = tenant_search_registry(store, tid)
                out = Agent(reg).run(QUESTION).output
                console.print(f"[bold]{tid}[/]: {out}\n")
        else:
            console.print("\n(set ANTHROPIC_API_KEY to also run the agent per tenant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
