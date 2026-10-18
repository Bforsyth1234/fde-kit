"""Run the golden set through the agent and gate on pass rate.

    python -m evals.run_evals                     # real model, threshold from settings
    python -m evals.run_evals --provider mock     # offline: replays each case's scripted mock
    python -m evals.run_evals --limit 5 --json out.json

Exit code is 0 when pass rate >= threshold, 1 otherwise. That exit code is the CI gate.

Mock mode exists to test the *harness*, not the model: each case ships a `mock` script that the
MockProvider replays. If a case passes in mock mode and fails against the real model, the model
is the problem; if it fails in both, the check or the tool is.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from fde_kit.agent import Agent
from fde_kit.builtin_tools import registry
from fde_kit.config import get_settings
from fde_kit.llm import LLMProvider, MockProvider, make_provider
from fde_kit.tracing import init_tracing

from .scorers import SCORERS, JudgeContext

GOLDEN = Path(__file__).parent / "golden.jsonl"
console = Console()


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    output: str
    tools_called: list[str]
    turns: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    details: list[str]


def load_cases(
    path: Path = GOLDEN, limit: int | None = None, ids: set[str] | None = None
) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for c in cases:
        for key in ("id", "prompt", "checks"):
            if key not in c:
                raise ValueError(f"golden case missing '{key}': {c}")
        for chk in c["checks"]:
            if chk["type"] not in SCORERS:
                raise ValueError(f"{c['id']}: unknown check type {chk['type']!r}")
    if ids:
        missing = ids - {c["id"] for c in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
        cases = [c for c in cases if c["id"] in ids]
    return cases[:limit] if limit else cases


def run_case(case: dict[str, Any], provider_name: str, judge: LLMProvider | None) -> CaseResult:
    settings = get_settings()
    if provider_name == "mock":
        provider: LLMProvider = MockProvider(case.get("mock", []))
    else:
        provider = make_provider(
            provider_name,
            settings.model,
            api_key=settings.anthropic_api_key,
            workspace_id=settings.anthropic_workspace_id,
        )

    agent = Agent(registry, provider=provider, settings=settings)
    t0 = time.perf_counter()
    result = agent.run(case["prompt"])
    latency = (time.perf_counter() - t0) * 1000

    ctx = JudgeContext(provider=judge, prompt=case["prompt"])
    details: list[str] = []
    passed = True
    for chk in case["checks"]:
        outcome = SCORERS[chk["type"]](result, chk.get("value"), ctx)
        details.append(("PASS " if outcome.passed else "FAIL ") + outcome.detail)
        passed = passed and outcome.passed

    return CaseResult(
        id=case["id"],
        category=case.get("category", ""),
        passed=passed,
        output=result.output,
        tools_called=result.tools_called,
        turns=result.turns,
        latency_ms=latency,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        details=details,
    )


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    by_cat: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    return {
        "total": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": (sum(r.passed for r in results) / len(results)) if results else 0.0,
        "p50_latency_ms": _percentile([r.latency_ms for r in results], 50),
        "p95_latency_ms": _percentile([r.latency_ms for r in results], 95),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "by_category": {
            cat: {"total": len(rs), "passed": sum(r.passed for r in rs)}
            for cat, rs in sorted(by_cat.items())
        },
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((pct / 100) * (len(s) - 1))))
    return s[k]


def print_report(results: list[CaseResult], summary: dict[str, Any], threshold: float) -> None:
    table = Table(title="eval results", show_lines=False)
    table.add_column("id")
    table.add_column("cat")
    table.add_column("ok", justify="center")
    table.add_column("tools")
    table.add_column("turns", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("failing checks")
    for r in results:
        fails = "; ".join(d for d in r.details if d.startswith("FAIL"))
        table.add_row(
            r.id,
            r.category,
            "[green]✓[/]" if r.passed else "[red]✗[/]",
            ",".join(r.tools_called) or "-",
            str(r.turns),
            f"{r.latency_ms:.0f}",
            fails,
        )
    console.print(table)

    cat_line = "  ".join(
        f"{c}={v['passed']}/{v['total']}" for c, v in summary["by_category"].items()
    )
    console.print(cat_line)
    verdict = "PASS" if summary["pass_rate"] >= threshold else "FAIL"
    color = "green" if verdict == "PASS" else "red"
    console.print(
        f"[{color} bold]{verdict}[/] {summary['passed']}/{summary['total']} "
        f"({summary['pass_rate']:.0%}) threshold {threshold:.0%} | "
        f"p50 {summary['p50_latency_ms']:.0f} ms, p95 {summary['p95_latency_ms']:.0f} ms | "
        f"tokens {summary['input_tokens']} in / {summary['output_tokens']} out"
    )


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="fde-kit eval runner")
    p.add_argument("--provider", default=settings.provider, choices=["anthropic", "mock"])
    p.add_argument("--threshold", type=float, default=settings.eval_pass_threshold)
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="comma-separated case ids to run, e.g. calc-02,notool-01")
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--json", type=Path, help="write full results to this file")
    p.add_argument(
        "--no-judge", action="store_true", help="skip LLM-as-judge checks even with a real provider"
    )
    args = p.parse_args(argv)

    # Mock runs are a harness self-check; there is nothing worth tracing and no collector
    # is expected to be running.
    if args.provider != "mock":
        init_tracing(settings)
    ids = {i.strip() for i in args.ids.split(",") if i.strip()} if args.ids else None
    cases = load_cases(args.golden, args.limit, ids)

    judge: LLMProvider | None = None
    if args.provider != "mock" and not args.no_judge:
        judge = make_provider(
            args.provider,
            settings.model,
            api_key=settings.anthropic_api_key,
            workspace_id=settings.anthropic_workspace_id,
        )

    results = [run_case(c, args.provider, judge) for c in cases]
    summary = summarise(results)
    print_report(results, summary, args.threshold)

    if args.json:
        args.json.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]}, indent=2
            )
        )
        console.print(f"wrote {args.json}")

    return 0 if summary["pass_rate"] >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
