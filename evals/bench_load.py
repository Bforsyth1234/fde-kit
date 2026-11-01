"""Time a wrangle load into Postgres, stage by stage.

    python -m evals.bench_load --admin-url postgresql://fde@localhost:5432/fde \
        [--spec tests/fixtures/wrangle/real/project/specs/mrts_all_nsa.yaml] \
        [--file tests/fixtures/wrangle/real/mrtssales92-present.xlsx] [--runs 1]

Loads the spec's files into a fresh tenant (dropped and re-created each run) with the hash
embedder, and reports wall time for: parse (`apply.run`), documents (`ingest_versioned`: store,
chunk, embed), facts (`add_facts`), and the whole `apply_to_tenant`. The default spec is the
committed 35-sheet Census retail load, 25,207 real facts in 35 documents, so the numbers are
reproducible from the repo alone.

The spec is copied to a temp dir and approved there; the committed files are not modified.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "fixtures" / "wrangle" / "real"
DEFAULT_SPEC = REAL / "project" / "specs" / "mrts_all_nsa.yaml"
DEFAULT_FILE = REAL / "mrtssales92-present.xlsx"
TENANT = "bench-load"


@contextmanager
def _timed(totals: dict[str, float], key: str, obj: object, name: str):
    """Wrap `obj.name` so every call adds its wall time to totals[key]."""
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            totals[key] += time.perf_counter() - t

    setattr(obj, name, wrapper)
    try:
        yield
    finally:
        setattr(obj, name, orig)


def bench(admin_url: str, spec: Path, files: list[Path]) -> dict[str, float]:
    from fde_kit.rag import HashEmbedder, TenantStore
    from fde_kit.rag.schema import app_url_from_admin, create_tenant, drop_tenant, migrate
    from fde_kit.rag.store import TenantSession
    from fde_kit.wrangle import adapter as wadapter
    from fde_kit.wrangle import apply as wapply
    from fde_kit.wrangle.spec import approve, load_project_spec

    migrate(admin_url)
    drop_tenant(admin_url, TENANT)
    create_tenant(admin_url, TENANT)

    tmp = Path(tempfile.mkdtemp())
    project = tmp / "project"
    shutil.copytree(spec.parent.parent, project)
    spec_copy = project / spec.parent.name / spec.name
    approve(spec_copy, by="bench")
    loaded = load_project_spec(spec_copy)

    totals = {"parse": 0.0, "documents": 0.0, "facts": 0.0}
    t0 = time.perf_counter()
    with (
        _timed(totals, "parse", wadapter, "run"),
        _timed(totals, "documents", TenantSession, "ingest_versioned"),
        _timed(totals, "facts", TenantSession, "add_facts"),
        TenantStore(app_url_from_admin(admin_url), HashEmbedder()) as store,
        store.session(TENANT) as s,
    ):
        result, report = wadapter.apply_to_tenant(s, loaded, files)
    totals["total"] = time.perf_counter() - t0
    assert report is not None, wapply.gate(result, approved=result.approved)
    totals["facts_written"] = report.facts
    totals["documents_written"] = report.documents_changed
    drop_tenant(admin_url, TENANT)
    shutil.rmtree(tmp)
    return totals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--admin-url", required=True)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--file", type=Path, action="append")
    p.add_argument("--runs", type=int, default=1)
    args = p.parse_args()
    files = args.file or [DEFAULT_FILE]
    for i in range(args.runs):
        t = bench(args.admin_url, args.spec, files)
        per_fact_ms = 1000 * t["facts"] / max(t["facts_written"], 1)
        print(
            f"run {i + 1}: {int(t['facts_written'])} facts, {int(t['documents_written'])} docs | "
            f"total {t['total']:.1f}s = parse {t['parse']:.1f}s + documents "
            f"{t['documents']:.1f}s + facts {t['facts']:.1f}s ({per_fact_ms:.2f} ms/fact)",
            flush=True,
        )


if __name__ == "__main__":
    main()
