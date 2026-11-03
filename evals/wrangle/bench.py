"""The wrangle benchmark: generate a world, map it, load it, score it (§5).

    fde wrangle bench [--seed 11] [--mapper heuristic|truth|agent] [--json out.json]
                      [--gate evals/wrangle/baseline.json] [--write-baseline PATH]

No database and no LLM. `--mapper truth` is the scorers' self-check: it exits 1 below 100%.
`--gate` exits 1 when any gated metric is worse than the committed baseline; CI runs it at a fixed
seed.
`--mapper agent` arrives with PR 7.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from fde_kit.wrangle.apply import run
from fde_kit.wrangle.spec import load_project_spec

from .generator import generate
from .mapper import HeuristicMapper, Mapper, TruthMapper
from .scorers import Loaded, score_all

DEFAULT_SEED = 11
"""CI's seed. Its header wording leaves every source loadable by the baseline, so no metric is
stuck at 0 and each one can regress."""
DEFAULT_BASELINE = Path(__file__).parent / "baseline.json"
LOWER_IS_BETTER = {"false_merges", "entity_misses", "phantom_facts"}
"""Gated metrics where a rise is the regression. Counts of what loaded are not gated at all."""
NOT_GATED = {"facts_loaded", "conflicts_flagged", "rejects_total"}


def run_mapper(mapper: Mapper, world: Path, project_dir: Path) -> tuple[list[Loaded], float]:
    """Propose, then load each spec with `apply.run`. A spec that can't run is recorded and
    scores nothing; it doesn't stop the bench."""
    start = time.perf_counter()
    loads = []
    for p in mapper.propose(world, project_dir):
        try:
            result = run(load_project_spec(p.spec_path), [p.file])
            loads.append(Loaded(p.source, result, file=p.file))
        except ValueError as e:  # SpecError, SpecMismatch and pydantic's ValidationError
            loads.append(Loaded(p.source, None, error=f"{p.spec_path.name}: {e}", file=p.file))
    return loads, time.perf_counter() - start


def bench(seed: int, mapper: Mapper, workdir: Path) -> dict:
    world = generate(seed, workdir / "world")
    loads, seconds = run_mapper(mapper, world, workdir / "proposed")
    scored = score_all(world, loads, workdir / "proposed")
    return {
        "seed": seed,
        "mapper": mapper.name,
        **scored,
        "cost": {"llm_calls": 0, "tokens": 0, "wall_seconds": round(seconds, 2)},
        "errors": [x.error for x in loads if x.error],
    }


def self_check(result: dict) -> list[str]:
    """`--mapper truth` must be perfect: anything less is a bug in the generator or a scorer."""
    return [
        f"{name}: {value}"
        for name, value in result["metrics"].items()
        if name not in NOT_GATED and value != (0 if name in LOWER_IS_BETTER else 1.0)
    ] + result["errors"]


def check_gate(result: dict, baseline: dict) -> list[str]:
    """What got worse than the baseline; empty when nothing did."""
    if (result["seed"], result["mapper"]) != (baseline["seed"], baseline["mapper"]):
        return [
            f"baseline is {baseline['mapper']} at seed {baseline['seed']}, "
            f"this run is {result['mapper']} at seed {result['seed']}"
        ]
    problems = []
    for name, floor in baseline["metrics"].items():
        got = result["metrics"].get(name)
        if name in NOT_GATED:
            continue
        worse = got is None or (got > floor if name in LOWER_IS_BETTER else got < floor)
        if worse:
            problems.append(f"{name}: {got} (baseline {floor})")
    return problems


def print_table(result: dict) -> None:
    print(f"wrangle bench: mapper={result['mapper']} seed={result['seed']}")
    for title, rows in (
        ("metrics", result["metrics"]),
        ("value accuracy by source", result["by_source"]),
        ("value accuracy by mess type", result["by_mess"]),
    ):
        print(f"\n{title}")
        for name, value in rows.items():
            print(f"  {name:<28} {value}")
    for x in result["false_merge_examples"]:
        print(f"  false merge: {x['raw']!r} -> {x['got']!r} (is {x['is']!r})")
    cost = result["cost"]
    print(f"\ncost: {cost['llm_calls']} LLM calls, {cost['wall_seconds']} s")
    for e in result["errors"]:
        print(f"spec error: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fde wrangle bench", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--mapper", choices=["heuristic", "truth", "agent"], default="heuristic")
    ap.add_argument("--provider", choices=["mock", "live"], help="for --mapper agent (PR 7)")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--gate", nargs="?", const=str(DEFAULT_BASELINE), help="baseline to gate on")
    ap.add_argument("--write-baseline", help="write this run's metrics as a baseline file")
    args = ap.parse_args(argv)

    if args.mapper == "agent":
        print("--mapper agent arrives with PR 7 (docs/wrangle-plan.md §6)", file=sys.stderr)
        return 2
    mapper: Mapper = TruthMapper() if args.mapper == "truth" else HeuristicMapper()
    with tempfile.TemporaryDirectory() as tmp:
        result = bench(args.seed, mapper, Path(tmp))
    print_table(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.write_baseline:
        baseline = {"seed": result["seed"], "mapper": result["mapper"]}
        baseline["metrics"] = result["metrics"]
        Path(args.write_baseline).write_text(json.dumps(baseline, indent=2) + "\n", "utf-8")
    if args.mapper == "truth":
        problems = self_check(result)
        if problems:
            print("\nself-check failed:", *problems, sep="\n  ")
            return 1
        print("\nself-check: the correct project scores 100%")
    if args.gate:
        problems = check_gate(result, json.loads(Path(args.gate).read_text(encoding="utf-8")))
        if problems:
            print("\nbenchmark regressed:", *problems, sep="\n  ")
            return 1
        print("\ngate: no regression against", args.gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
