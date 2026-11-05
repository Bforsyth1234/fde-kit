"""The wrangle benchmark: generate a world, map it, load it, score it (§5).

    fde wrangle bench [--seed N ...] [--mapper heuristic|truth|agent] [--json out.json]
                      [--gate [evals/wrangle/baseline.json]] [--write-baseline PATH]

No database and no LLM. `--mapper truth` is the scorers' self-check: it exits 1 below 100%.
`--gate` exits 1 when any gated metric is worse than the committed baseline, at every seed the
baseline holds unless `--seed` names some; CI runs it. An improvement passes the gate and is
printed, so the floor can be re-recorded with `--write-baseline`, which only writes after every
check passed. `--mapper agent` arrives with PR 7.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fde_kit.wrangle.apply import run
from fde_kit.wrangle.spec import load_project_spec

from .generator import generate
from .mapper import HeuristicMapper, Mapper, TruthMapper
from .scorers import Loaded, score_all

DEFAULT_SEED = 11
"""The seed a run without `--seed` or `--gate` uses. Its header wording leaves every source
loadable by the baseline, so no metric is stuck at 0."""
DEFAULT_BASELINE = Path(__file__).parent / "baseline.json"
LOWER_IS_BETTER = {"false_merges", "entity_misses", "phantom_facts"}
"""Gated metrics where a rise is the regression. Counts of what loaded are not gated at all."""
NOT_GATED = {"facts_loaded", "conflicts_flagged", "rejects_total"}
SCORING_PACKAGES = ("rapidfuzz", "openpyxl")
"""Packages whose releases can move the baseline's scores with no change here (rapidfuzz's
ratios decide every fuzzy match). The baseline records their versions, and the gate says when
they differ, so a moved score is traced to its cause."""


def package_versions() -> dict[str, str]:
    out = {}
    for name in SCORING_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "missing"
    return out


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


def _worse(name: str, got, floor) -> bool:
    return got is None or (got > floor if name in LOWER_IS_BETTER else got < floor)


def check_gate(result: dict, baseline: dict) -> list[str]:
    """What got worse than the baseline at this run's seed; empty when nothing did."""
    floors = baseline["seeds"].get(str(result["seed"]))
    if result["mapper"] != baseline["mapper"] or floors is None:
        return [
            f"baseline is {baseline['mapper']} at seeds {', '.join(baseline['seeds'])}, "
            f"this run is {result['mapper']} at seed {result['seed']}"
        ]
    return [
        f"{name}: {result['metrics'].get(name)} (baseline {floor})"
        for name, floor in floors.items()
        if name not in NOT_GATED and _worse(name, result["metrics"].get(name), floor)
    ]


def improvements(result: dict, baseline: dict) -> list[str]:
    """Gated metrics that beat the baseline: passing, but worth re-recording the floor."""
    floors = baseline["seeds"].get(str(result["seed"]), {})
    return [
        f"{name}: {result['metrics'][name]} (baseline {floor})"
        for name, floor in floors.items()
        if name not in NOT_GATED
        and name in result["metrics"]
        and result["metrics"][name] != floor
        and not _worse(name, result["metrics"][name], floor)
    ]


def write_baseline(path: Path, results: list[dict]) -> None:
    """Record these runs' metrics, keeping seeds already in the file for the same mapper."""
    mapper = results[0]["mapper"]
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    seeds = old.get("seeds", {}) if old.get("mapper") == mapper else {}
    seeds.update({str(r["seed"]): r["metrics"] for r in results})
    baseline = {
        "mapper": mapper,
        "versions": package_versions(),
        "seeds": dict(sorted(seeds.items(), key=lambda kv: int(kv[0]))),
    }
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


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


def _seeds(args: argparse.Namespace, baseline: dict | None) -> list[int]:
    if args.seed:
        return list(dict.fromkeys(args.seed))
    if baseline is not None:
        return [int(k) for k in baseline["seeds"]]
    return [DEFAULT_SEED]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fde wrangle bench", description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, nargs="+", help=f"default: {DEFAULT_SEED}, or the gate's")
    ap.add_argument("--mapper", choices=["heuristic", "truth", "agent"], default="heuristic")
    ap.add_argument("--provider", choices=["mock", "live"], help="for --mapper agent (PR 7)")
    ap.add_argument("--json", help="write the full result here (a list with several seeds)")
    ap.add_argument("--gate", nargs="?", const=str(DEFAULT_BASELINE), help="baseline to gate on")
    ap.add_argument("--write-baseline", help="record this run's metrics, once every check passed")
    args = ap.parse_args(argv)

    if args.mapper == "agent":
        print("--mapper agent arrives with PR 7 (docs/wrangle-plan.md §6)", file=sys.stderr)
        return 2
    if args.provider:
        print(f"--provider is for --mapper agent; ignored for {args.mapper}", file=sys.stderr)
    if args.gate and args.write_baseline:
        if Path(args.gate).resolve() == Path(args.write_baseline).resolve():
            print("--gate and --write-baseline name the same file", file=sys.stderr)
            return 2
    baseline = json.loads(Path(args.gate).read_text(encoding="utf-8")) if args.gate else None
    mapper: Mapper = TruthMapper() if args.mapper == "truth" else HeuristicMapper()

    results, failed = [], False
    for seed in _seeds(args, baseline):
        with tempfile.TemporaryDirectory() as tmp:
            result = bench(seed, mapper, Path(tmp))
        results.append(result)
        print_table(result)
        if args.mapper == "truth":
            problems = self_check(result)
            if problems:
                print("\nself-check failed:", *problems, sep="\n  ")
                failed = True
            else:
                print("\nself-check: the correct project scores 100%")
        if baseline is not None:
            problems = check_gate(result, baseline)
            if problems:
                print("\nbenchmark regressed:", *problems, sep="\n  ")
                failed = True
            else:
                print(f"\ngate: no regression against {args.gate} at seed {seed}")
            better = improvements(result, baseline)
            if better:
                print("improved (re-record with --write-baseline):", *better, sep="\n  ")
        print()
    if baseline is not None and baseline.get("versions", {}) != package_versions():
        print(
            f"note: baseline recorded with {baseline.get('versions')}, "
            f"installed {package_versions()}; a moved fuzzy score may come from that"
        )
    if args.json:
        out = results[0] if len(results) == 1 else results
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if failed:
        return 1
    if args.write_baseline:
        write_baseline(Path(args.write_baseline), results)
        print("baseline written:", args.write_baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
