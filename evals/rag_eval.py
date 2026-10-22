"""Per-tenant retrieval eval over the demo lease corpus.

    python -m evals.rag_eval                    # bge-small embeddings (downloads ~130 MB once)
    python -m evals.rag_eval --embedder hash    # offline, deterministic; what CI runs
    python -m evals.rag_eval --json rag-results.json

For every tenant and every search mode it reports (hybrid = relative score fusion,
hybrid-rrf = reciprocal rank fusion, k=60):

- hit@1 / hit@k: share of questions where a relevant chunk is first / in the top k. Relevant
  means the chunk contains the expected answer string and comes from an expected document.
- MRR: mean reciprocal rank of the first relevant chunk.
- p50 latency per query.
- cross-tenant hits: chunks in any result set whose source belongs to another tenant. This is
  measured, not assumed, and includes "probe" questions where a tenant asks about another
  tenant's documents (Summit asking for Oakline's pool hours). It must be zero.

Exit code 1 if cross-tenant hits > 0 or hybrid hit@k for any tenant is below --threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.table import Table

from fde_kit.rag import TenantStore, app_url_from_admin, create_tenant, make_embedder, migrate
from fde_kit.rag.demo_corpus import TENANTS, Question, questions, seed

MODES = ("vector", "keyword", "hybrid-rrf", "hybrid")
_SEARCH = {
    "vector": {"mode": "vector"},
    "keyword": {"mode": "keyword"},
    "hybrid-rrf": {"mode": "hybrid", "fusion": "rrf"},
    "hybrid": {"mode": "hybrid", "fusion": "relative"},
}
console = Console()


@dataclass
class Row:
    tenant: str
    mode: str
    n: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    p50_ms: float
    cross_tenant_hits: int


def _relevant(q: Question, source: str, content: str) -> bool:
    return q.answer in content and (not q.sources or source in q.sources)


def run(admin_url: str, embedder_name: str, k: int) -> tuple[list[Row], list[dict]]:
    migrate(admin_url)
    for tid, t in TENANTS.items():
        create_tenant(admin_url, tid, t["name"])
    embedder = make_embedder(embedder_name)
    rows: list[Row] = []
    detail: list[dict] = []
    with TenantStore(app_url_from_admin(admin_url), embedder) as store:
        counts = seed(store)
        console.print(f"seeded: {counts}  embedder: {embedder.name}")
        qs = questions()
        for tid in TENANTS:
            own = [q for q in qs if q.tenant == tid]
            # Probes: other tenants' questions, asked as this tenant. Every hit must be ours.
            probes = [q for q in qs if q.tenant != tid and q.category == "tenant-only"]
            for mode in MODES:
                hits_at_1, hits_at_k, rr, lat, leaks = [], [], [], [], 0
                with store.session(tid) as s:
                    for q in own + probes:
                        t0 = time.perf_counter()
                        hits = s.search(q.query, k=k, **_SEARCH[mode])
                        lat.append((time.perf_counter() - t0) * 1000)
                        leaks += sum(1 for h in hits if not h.source.startswith(f"{tid}/"))
                        if q.tenant != tid:
                            continue
                        rank = next(
                            (i for i, h in enumerate(hits, 1) if _relevant(q, h.source, h.content)),
                            None,
                        )
                        hits_at_1.append(rank == 1)
                        hits_at_k.append(rank is not None)
                        rr.append(1 / rank if rank else 0.0)
                        detail.append(
                            {
                                "tenant": tid,
                                "mode": mode,
                                "id": q.id,
                                "category": q.category,
                                "query": q.query,
                                "rank": rank,
                                "top": [h.source for h in hits],
                            }
                        )
                rows.append(
                    Row(
                        tid,
                        mode,
                        len(own),
                        sum(hits_at_1) / len(own),
                        sum(hits_at_k) / len(own),
                        statistics.mean(rr),
                        statistics.median(lat),
                        leaks,
                    )
                )
    return rows, detail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--admin-url",
        default=os.environ.get("FDE_RAG_ADMIN_URL", "postgresql://fde:fde@localhost:5432/fde"),
    )
    p.add_argument("--embedder", default=os.environ.get("FDE_RAG_EMBEDDER", "bge"))
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.0, help="min hybrid hit@k per tenant")
    p.add_argument("--json", help="write per-question results here")
    a = p.parse_args(argv)

    rows, detail = run(a.admin_url, a.embedder, a.k)

    table = Table(title=f"retrieval, k={a.k}, embedder={a.embedder}")
    for col in (
        "tenant",
        "mode",
        "questions",
        "hit@1",
        f"hit@{a.k}",
        "MRR",
        "p50 ms",
        "cross-tenant",
    ):
        table.add_column(col, justify="left" if col in ("tenant", "mode") else "right")
    for r in rows:
        table.add_row(
            r.tenant,
            r.mode,
            str(r.n),
            f"{r.hit_at_1:.2f}",
            f"{r.hit_at_k:.2f}",
            f"{r.mrr:.2f}",
            f"{r.p50_ms:.1f}",
            str(r.cross_tenant_hits),
        )
    console.print(table)

    cats = sorted({d["category"] for d in detail})
    ct = Table(title=f"hit@1 by question category (all tenants), embedder={a.embedder}")
    ct.add_column("category")
    ct.add_column("questions", justify="right")
    for mode in MODES:
        ct.add_column(mode, justify="right")
    for c in cats:
        cells = []
        for mode in MODES:
            rs = [d["rank"] for d in detail if d["category"] == c and d["mode"] == mode]
            cells.append(f"{sum(r == 1 for r in rs) / len(rs):.2f}")
        n = sum(1 for d in detail if d["category"] == c and d["mode"] == MODES[0])
        ct.add_row(c, str(n), *cells)
    console.print(ct)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": [asdict(r) for r in rows], "questions": detail}, f, indent=2)

    leaks = sum(r.cross_tenant_hits for r in rows)
    low = [r for r in rows if r.mode == "hybrid" and r.hit_at_k < a.threshold]
    if leaks:
        console.print(f"[red]FAIL: {leaks} cross-tenant hits[/]")
    for r in low:
        console.print(f"[red]FAIL: {r.tenant} hybrid hit@{a.k} {r.hit_at_k:.2f} < {a.threshold}[/]")
    return 1 if leaks or low else 0


if __name__ == "__main__":
    sys.exit(main())
