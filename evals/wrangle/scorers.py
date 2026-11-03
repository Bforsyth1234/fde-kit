"""Score a mapper's load against a world's truth (docs/wrangle-plan.md §5.2).

Everything here reads `truth/` and the `ApplyResult`s; nothing is written anywhere. A truth fact
is keyed by (source, entity, field, period_start). Values are compared with `values_agree` at the
truth's precision, not the mapper's, so a mapper can't pass by declaring a coarser one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from fde_kit.rag.store import values_agree
from fde_kit.wrangle.apply import ApplyResult, FactRecord, cell_pipeline, parse_source_ref
from fde_kit.wrangle.parse import to_fact_value
from fde_kit.wrangle.read import parse_ref, read_grid
from fde_kit.wrangle.resolve import Resolver
from fde_kit.wrangle.spec import Crosswalk, _load_crosswalk

from .generator import UNPIVOT

FALSE_MERGE_WEIGHT = 5


@dataclass
class Loaded:
    """One spec's load: its source, and the run's result (None when the spec didn't run)."""

    source: str
    result: ApplyResult | None
    error: str = ""
    file: Path | None = None


@dataclass
class TruthWorld:
    facts: list[dict] = field(default_factory=list)
    messes: dict[str, str] = field(default_factory=dict)
    rejects: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    columns: list[dict] = field(default_factory=list)
    coa: dict[str, str] = field(default_factory=dict)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_truth(world: Path) -> TruthWorld:
    t = world / "truth"
    return TruthWorld(
        facts=_jsonl(t / "facts.jsonl"),
        messes={m["id"]: m["type"] for m in _jsonl(t / "messes.jsonl")},
        rejects=_jsonl(t / "rejects.jsonl"),
        conflicts=_jsonl(t / "conflicts.jsonl"),
        entities=_jsonl(t / "entities.jsonl"),
        columns=_jsonl(t / "columns.jsonl"),
        coa=json.loads((t / "coa.json").read_text(encoding="utf-8")),
    )


def _ratio(n: float, d: float, empty: float = 1.0) -> float:
    """n/d rounded for a stable JSON; `empty` when there is nothing to divide by."""
    return round(n / d, 6) if d else empty


def _f1_parts(hits: int, proposed: int, wanted: int) -> tuple[float, float]:
    """Precision and recall. Proposing nothing earns no precision: a load that flags no conflict
    at all shouldn't read as 1.0 and then gate against a mapper that finds some."""
    return _ratio(hits, proposed, empty=0.0), _ratio(hits, wanted)


def _records(loads: list[Loaded]):
    for load in loads:
        if load.result is None:
            continue
        for f in load.result.files:
            for rec in f.facts:
                yield load, rec


def _key(source: str, rec: FactRecord) -> tuple[str, str, str, str]:
    row = rec.fact.row
    return (source, row.entity, row.field, row.period_start.isoformat())


def score_columns(truth: TruthWorld, loads: list[Loaded]) -> dict:
    """Column mapping P/R: (source, column, field, scale, unit) must all be right."""
    wanted = {(c["source"], c["column"], c["field"], c["scale"], c["unit"]) for c in truth.columns}
    proposed = set()
    for load in loads:
        if load.result is None:
            continue
        loaded = load.result.loaded
        for m in loaded.spec.measures:
            scale = int(m.scale) if float(m.scale).is_integer() else m.scale
            proposed.add((load.source, m.column or UNPIVOT, m.field, scale, loaded.unit_for(m)))
    p, r = _f1_parts(len(wanted & proposed), len(proposed), len(wanted))
    return {"column_mapping_precision": p, "column_mapping_recall": r}


def _crosswalk(project_dir: Path, name: str) -> Crosswalk | None:
    """Read from the project, not from a load: a spec that failed to run still proposed it."""
    path = project_dir / "crosswalks" / f"{name}.csv"
    return _load_crosswalk(name, name, path, path.read_bytes()) if path.exists() else None


def score_entities(truth: TruthWorld, project_dir: Path) -> tuple[dict, list[dict]]:
    """Each distinct raw name in the sources, through the mapper's crosswalk with the real
    `Resolver`. A wrong canonical (or a name that should stay unknown getting one) is a false
    merge: reported on its own and weighted ×5 in precision. A name left unknown is a miss."""
    crosswalk = _crosswalk(project_dir, "plants")
    resolver = Resolver(crosswalk=crosswalk, policy="closed") if crosswalk else None
    tp = false_merges = misses = 0
    merged: list[dict] = []
    for e in truth.entities:
        got = resolver.resolve(e["raw"]).canonical if resolver else None
        if got is None:
            misses += e["canonical"] is not None
        elif got == e["canonical"]:
            tp += 1
        else:
            false_merges += 1
            merged.append(
                {"source": e["source"], "raw": e["raw"], "got": got, "is": e["canonical"]}
            )
    wanted = sum(e["canonical"] is not None for e in truth.entities)
    return {
        "entity_precision": _ratio(tp, tp + FALSE_MERGE_WEIGHT * false_merges, empty=0.0),
        "entity_recall": _ratio(tp, wanted),
        "false_merges": false_merges,
        "entity_misses": misses,
    }, merged


def score_coa(truth: TruthWorld, project_dir: Path) -> dict:
    """Local account code -> standard account, read from the mapper's `coa` crosswalk."""
    crosswalk = _crosswalk(project_dir, "coa")
    mapping = crosswalk.mapping if crosswalk else {}
    hits = sum(mapping.get(code) == standard for code, standard in truth.coa.items())
    return {"coa_accuracy": _ratio(hits, len(truth.coa))}


def score_facts(truth: TruthWorld, loads: list[Loaded]) -> tuple[dict, dict]:
    """Value accuracy (of the truth's facts, how many loaded right), value precision (of what
    loaded, how much is true), both overall, by source and by mess type."""
    loaded: dict[tuple, list[FactRecord]] = defaultdict(list)
    for load, rec in _records(loads):
        loaded[_key(load.source, rec)].append(rec)

    right: set[tuple] = set()
    per_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_mess: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    truth_keys = set()
    for f in truth.facts:
        key = (f["source"], f["entity"], f["field"], f["period_start"])
        truth_keys.add(key)
        recs = loaded.get(key, [])
        ok = any(
            values_agree(Decimal(f["value"]), f["precision"], r.fact.row.value, f["precision"])
            for r in recs
        )
        if ok:
            right.add(key)
        per_source[f["source"]][0] += ok
        per_source[f["source"]][1] += 1
        for mess_type in {truth.messes[m] for m in f["messes"]}:
            per_mess[mess_type][0] += ok
            per_mess[mess_type][1] += 1
    n_loaded = sum(len(v) for v in loaded.values())
    phantom = sum(len(v) for k, v in loaded.items() if k not in truth_keys)
    return {
        "value_accuracy": _ratio(len(right), len(truth.facts)),
        "value_precision": _ratio(len(right), n_loaded, empty=0.0),
        "facts_loaded": n_loaded,
        "phantom_facts": phantom,
    }, {
        "by_source": {k: _ratio(*v) for k, v in sorted(per_source.items())},
        "by_mess": {k: _ratio(*v) for k, v in sorted(per_mess.items())},
    }


def score_provenance(truth: TruthWorld, loads: list[Loaded]) -> dict:
    """§4.7 over the whole world: each loaded fact's `source_ref`, read back through the spec's
    own ops, gives that fact; and when the truth knows the fact, it is the truth's cell."""
    cells = {
        (f["source"], f["entity"], f["field"], f["period_start"]): f["cell"] for f in truth.facts
    }
    grids: dict[tuple, object] = {}
    good = total = 0
    for load, rec in _records(loads):
        total += 1
        loaded = load.result.loaded
        row = rec.fact.row
        ref = parse_source_ref(row.source_ref)
        measure = next(m for m in loaded.spec.measures if m.field == row.field)
        grid_key = (str(load.file), ref["sheet"], loaded.spec.source.delimiter)
        if grid_key not in grids:
            sheet = ref["sheet"] if load.file.suffix == ".xlsx" else None
            grids[grid_key] = read_grid(
                load.file,
                sheet,
                encoding=loaded.spec.source.encoding,
                delimiter=loaded.spec.source.delimiter,
            )
        cell = grids[grid_key].cell(*parse_ref(ref["cell"]))
        value = to_fact_value(cell_pipeline(measure.ops)(cell) * measure.scale_decimal)
        expected_cell = cells.get(_key(load.source, rec))
        good += value == row.value and expected_cell in (None, rec.cell)
    return {"provenance_accuracy": _ratio(good, total, empty=0.0)}


def score_conflicts(truth: TruthWorld, loads: list[Loaded]) -> dict:
    """The conflicts `add_facts` would record for this load, by the same `values_agree` rule,
    against the injected ones. Pure: no database."""
    groups: dict[tuple, list] = defaultdict(list)
    for _, rec in _records(loads):
        row = rec.fact.row
        groups[row.entity, row.field, row.period_start.isoformat()].append(row)
    flagged = set()
    for key, rows in groups.items():
        if any(
            not values_agree(a.value, a.precision, b.value, b.precision)
            for i, a in enumerate(rows)
            for b in rows[i + 1 :]
        ):
            flagged.add(key)
    wanted = {(c["entity"], c["field"], c["period"]) for c in truth.conflicts}
    p, r = _f1_parts(len(flagged & wanted), len(flagged), len(wanted))
    return {"conflict_precision": p, "conflict_recall": r, "conflicts_flagged": len(flagged)}


def score_rejects(truth: TruthWorld, loads: list[Loaded]) -> dict:
    """Rejects are the cells that really can't load, and those cells are rejected."""

    def key(source: str, r: dict) -> tuple:
        return (source, r["cell"]) if r.get("cell") else (source, f"row {r['row']}")

    wanted = {key(r["source"], r) for r in truth.rejects}
    got = set()
    for load in loads:
        if load.result is not None:
            got |= {key(load.source, r) for r in load.result.rejects}
    p, r = _f1_parts(len(got & wanted), len(got), len(wanted))
    return {"reject_precision": p, "reject_recall": r, "rejects_total": len(got)}


def score_all(world: Path, loads: list[Loaded], project_dir: Path) -> dict:
    truth = load_truth(world)
    entities, merged = score_entities(truth, project_dir)
    facts, breakdown = score_facts(truth, loads)
    return {
        "metrics": {
            **score_columns(truth, loads),
            **entities,
            **score_coa(truth, project_dir),
            **facts,
            **score_provenance(truth, loads),
            **score_conflicts(truth, loads),
            **score_rejects(truth, loads),
        },
        **breakdown,
        "false_merge_examples": merged[:10],
    }
