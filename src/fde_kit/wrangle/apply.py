"""Apply a mapping spec to files: cells -> facts with cell-level provenance (§7.4).

`run(spec, files)` is pure: no database. It is the dry run, and the adapter (`adapter.py`)
loads what it returns, so a dry run and an apply can never disagree about a value.

For each file the spec's sheet is read into a `Grid`, then:

1. `header_fingerprint`, if the spec sets one, must match (`force=True` skips only this check).
2. Column labels are built from `header_rows` (`labels.header_labels`); a column the spec names
   that isn't there is a `SpecMismatch`, and the whole file is refused.
3. Each data row: `skip_rows` (counted, not rejected), then the entity parts are resolved and
   the period parsed. A failure there is one **row-level** reject and none of the row's measures
   load. A row with every cell empty is passed over silently.
4. Each measure cell: empty + `empty: skip` is counted; otherwise ops then `scale` -> a fact, or
   a **cell-level** reject.
5. The same (entity, field, period) twice in one load: an equal value is dropped (the first
   cell is kept); a different value loads too, and the report names every cell involved.

Every fact's `source_ref` is `<Sheet>!<A1>@wrangle:<spec>/<bundle_sha12>/<file_sha12>`, and
every sheet becomes one `RawDocument` whose text is the sheet as CSV with Excel row numbers.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..rag.adapters.base import RawDocument, RawEntity, RawFact
from ..rag.store import FactRow
from .labels import header_fingerprint, header_labels
from .parse import ParseError, apply_text, compile_ops, period_of, to_fact_value
from .read import Cell, Grid, col_letter, parse_ref, read_grid
from .resolve import STRATEGIES, FuzzyMatch, Resolver
from .spec import LoadedSpec, Measure, SkipBlank, SpecError, load_project_spec

__all__ = [
    "ApplyResult",
    "FactRecord",
    "FileResult",
    "SpecMismatch",
    "gate",
    "parse_source_ref",
    "reject_rate",
    "run",
    "write_outputs",
]

PREVIEW_FACTS = 50
_SOURCE_REF = re.compile(
    r"^(?P<sheet>.+)!(?P<cell>[A-Z]{1,3}[1-9][0-9]*)"
    r"@wrangle:(?P<spec>[a-z0-9_]+)/(?P<bundle>[0-9a-f]{12})/(?P<file>[0-9a-f]{12})$"
)
_READ_FAILURES = {"error": "error cell", "formula_no_value": "formula without cached value"}


class SpecMismatch(ValueError):
    """The file doesn't fit the spec (fingerprint, a missing column or sheet): the whole file is
    refused, rather than turned into rejects."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path


def parse_source_ref(ref: str) -> dict[str, str]:
    """`Plan!G42@wrangle:planner_forecast/<bundle12>/<file12>` -> {sheet, cell, spec, bundle,
    file}. ValueError for anything else."""
    m = _SOURCE_REF.match(ref)
    if not m:
        raise ValueError(f"not a wrangle source_ref: {ref!r}")
    return m.groupdict()


# --- results --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactRecord:
    """A fact with where it came from: the cell (also in `source_ref`) plus the cells that gave
    its entity and period, which the report and preview show but the ref doesn't carry."""

    fact: RawFact
    file: str
    cell: str
    """"Sheet!A1"."""
    entity_cells: tuple[str, ...]
    period_from: str
    """The period's cell ("B7"), or for an unpivoted cell its column label."""


@dataclass
class FileResult:
    path: str
    """As passed to `run`."""
    sheet: str
    file_sha256: str
    document: RawDocument
    facts: list[FactRecord] = field(default_factory=list)
    entities: list[RawEntity] = field(default_factory=list)
    """Composite and open entities with at least one fact in this file."""
    rows_read: int = 0
    rows_skipped: dict[str, int] = field(default_factory=dict)
    empty_skipped: int = 0


@dataclass
class ApplyResult:
    loaded: LoadedSpec
    files: list[FileResult]
    crosswalk_entities: list[RawEntity]
    rejects: list[dict[str, Any]]
    report: dict[str, Any]
    proposed: list[FuzzyMatch]
    duplicates_dropped: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.loaded.approved

    @property
    def facts(self) -> list[RawFact]:
        return [r.fact for f in self.files for r in f.facts]

    @property
    def documents(self) -> list[RawDocument]:
        return [f.document for f in self.files]

    @property
    def entities(self) -> list[RawEntity]:
        """Crosswalk entities first, then each composite/open entity once."""
        out = list(self.crosswalk_entities)
        seen = {(e.kind, e.canonical_name) for e in out}
        for f in self.files:
            for e in f.entities:
                if (e.kind, e.canonical_name) not in seen:
                    seen.add((e.kind, e.canonical_name))
                    out.append(e)
        return out

    @property
    def reject_rate(self) -> float:
        return reject_rate(len(self.facts), len(self.rejects))


def reject_rate(facts: int, rejects: int) -> float:
    """rejects / (facts + rejects); 0 when nothing was read. Rule and empty skips don't count."""
    total = facts + rejects
    return rejects / total if total else 0.0


# --- compiling the spec ---------------------------------------------------------------------

Pipeline = Callable[[str], object]


def _compile(ops: list[dict[str, Any]], where: str) -> Pipeline | None:
    if not ops:
        return None
    try:
        return compile_ops(ops)
    except ValueError as e:
        raise SpecError(f"{where}: {e}") from e


def _period_pipeline(ops: list[dict[str, Any]], where: str, grain: str | None) -> Pipeline:
    if not ops:
        raise SpecError(f"{where}: needs ops ending in `date` or `period`")
    last = ops[-1]["op"]
    if last not in ("date", "period"):
        raise SpecError(f"{where}: ops must end in `date` or `period`, not {last!r}")
    if last == "period" and grain is not None:
        raise SpecError(f"{where}: `grain` goes in the `period` op, not beside it")
    fn = _compile(ops, where)
    assert fn is not None

    def run(raw: str) -> tuple[date, date]:
        v = fn(raw)
        if isinstance(v, tuple):
            return v
        assert isinstance(v, date)
        return period_of(v, grain or "day")

    return run


@dataclass
class _Compiled:
    parts: list[tuple[Pipeline | None, Resolver]]
    period_column: Pipeline | None
    period_cell: Pipeline | None
    unpivot: Pipeline | None
    measures: list[tuple[Measure, Pipeline | None]]
    skips: list[tuple[str, Any]]


def _compile_spec(loaded: LoadedSpec) -> _Compiled:
    spec = loaded.spec
    parts = [
        (
            _compile(p.ops, f"entity.parts[{i}]"),
            Resolver.for_part(p.resolve, loaded.crosswalks),
        )
        for i, p in enumerate(spec.entity.parts)
    ]
    per = spec.period
    measures = []
    for i, m in enumerate(spec.measures):
        if not m.is_text and m.ops[-1]["op"] != "number":
            raise SpecError(f"measures[{i}] ({m.field}): numeric measures end in `number`")
        measures.append((m, _compile(m.ops, f"measures[{i}]")))
    skips: list[tuple[str, Any]] = []
    for rule in spec.source.skip_rows:
        if isinstance(rule, SkipBlank):
            skips.append(("blank", rule))
        else:
            skips.append(("match", (rule, re.compile(rule.matches))))
    return _Compiled(
        parts=parts,
        period_column=(
            _period_pipeline(per.column.ops, "period.column", per.column.grain)
            if per.column
            else None
        ),
        period_cell=_period_pipeline(per.cell.ops, "period.cell", None) if per.cell else None,
        unpivot=_period_pipeline(per.unpivot.ops, "period.unpivot", None) if per.unpivot else None,
        measures=measures,
        skips=skips,
    )


# --- per file -------------------------------------------------------------------------------


class _Columns:
    """Joined header labels -> 1-based column numbers."""

    def __init__(self, path: str, labels: list[str]):
        self.path = path
        self.labels = labels
        self._exact: dict[str, list[int]] = {}
        self._loose: dict[str, list[int]] = {}
        for i, label in enumerate(labels, start=1):
            self._exact.setdefault(label, []).append(i)
            self._loose.setdefault(apply_text(label), []).append(i)

    def find(self, name: str, what: str) -> int:
        cols = self._exact.get(name) or self._loose.get(apply_text(name), [])
        if not cols:
            shown = [lbl for lbl in self.labels if lbl]
            raise SpecMismatch(self.path, f"{what} {name!r} is not a column; labels: {shown}")
        if len(cols) > 1:
            letters = [col_letter(c) for c in cols]
            raise SpecMismatch(self.path, f"{what} {name!r} names several columns: {letters}")
        return cols[0]


def _render_document(grid: Grid) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    width = len(grid.rows[0]) if grid.rows else 0
    w.writerow(["row", *(col_letter(c) for c in range(1, width + 1))])
    for i, row in enumerate(grid.rows, start=1):
        w.writerow([i, *(c.raw for c in row)])
    return buf.getvalue()


def _period_label(start: date, end: date) -> str:
    if start == end:
        return start.isoformat()
    if start.day == 1 and period_of(start, "month") == (start, end):
        return start.strftime("%Y-%m")
    return f"{start.isoformat()}..{end.isoformat()}"


def _precision(m: Measure) -> int | float | None:
    if m.is_text:
        return None
    p = m.effective_precision
    if p is not None and float(p).is_integer():
        return int(p)
    return p


class _Run:
    def __init__(self, loaded: LoadedSpec, force: bool):
        self.loaded = loaded
        self.spec = loaded.spec
        self.force = force
        self.c = _compile_spec(loaded)
        self.rejects: list[dict[str, Any]] = []
        self.duplicates = 0
        # (entity, field, start, end) -> [(value, "Sheet!A1")]; values within a key all differ.
        self.seen: dict[tuple[str, str, date, date], list[tuple[Any, str]]] = {}
        self.crosswalk_canonicals = {c for cw in loaded.crosswalks.values() for c in cw.canonicals}

    def _reject(self, fr: FileResult, *, where: dict[str, Any], **kw: Any) -> None:
        rec: dict[str, Any] = {"file": fr.path, "sheet": fr.sheet, **where}
        rec.update(kw)
        self.rejects.append(rec)

    def _row_reject(self, fr, row: int, cell: Cell, stage: str, op: str, reason: str) -> None:
        self._reject(
            fr,
            where={"row": row, "level": "row"},
            raw=cell.raw,
            stage=stage,
            op=op,
            reason=reason,
        )

    def _cell_reject(self, fr, cell: Cell, raw: str, stage: str, op: str, reason: str) -> None:
        self._reject(
            fr,
            where={"cell": cell.ref, "level": "cell"},
            raw=raw,
            stage=stage,
            op=op,
            reason=reason,
        )

    def file(self, path: Path | str) -> FileResult:
        spec, src = self.spec, self.spec.source
        shown = str(path)
        try:
            grid = read_grid(
                path,
                src.sheet if Path(path).suffix.lower() in (".xlsx", ".xlsm") else None,
                encoding=src.encoding,
                delimiter=src.delimiter,
            )
        except ValueError as e:
            raise SpecMismatch(shown, str(e)) from e

        labels = header_labels(grid, list(src.header_rows))
        if src.header_fingerprint and not self.force:
            got = header_fingerprint(labels)
            if got != src.header_fingerprint:
                raise SpecMismatch(
                    shown,
                    f"header_fingerprint is {got}, the spec expects {src.header_fingerprint} "
                    "(the header changed; re-check the spec, or --force)",
                )
        cols = _Columns(shown, labels)
        part_cols = [cols.find(p.column, "entity column") for p in spec.entity.parts]
        skip_cols: list[tuple[str, Any]] = []
        for kind, rule in self.c.skips:
            if kind == "blank":
                skip_cols.append(
                    (kind, [cols.find(n, "skip_rows blank column") for n in rule.blank])
                )
            else:
                r, rx = rule
                skip_cols.append((kind, (cols.find(r.column, "skip_rows column"), rx)))
        per = spec.period
        period_col = cols.find(per.column.name, "period column") if per.column else None
        unpivot_cols: list[int] = []
        if per.unpivot is not None:
            if per.unpivot.columns is not None:
                lo = cols.find(per.unpivot.columns.from_, "unpivot.columns.from")
                hi = cols.find(per.unpivot.columns.to, "unpivot.columns.to")
                if lo > hi:
                    raise SpecMismatch(shown, "unpivot.columns.from is right of `to`")
                unpivot_cols = list(range(lo, hi + 1))
            else:
                rx = re.compile(per.unpivot.matching or "")
                unpivot_cols = [i for i, lbl in enumerate(labels, 1) if rx.search(lbl)]
                if not unpivot_cols:
                    raise SpecMismatch(shown, f"no column label matches {per.unpivot.matching!r}")
        measure_cols = [
            (m, fn, cols.find(m.column, "measure column") if m.column is not None else None)
            for m, fn in self.c.measures
        ]

        source = spec.source.document_key.format(path=shown, sheet=grid.sheet)
        document = RawDocument(
            source=source,
            title=f"{Path(path).name} — {grid.sheet}",
            text=_render_document(grid),
            metadata={
                "file_sha256": grid.file_sha256,
                "spec": spec.spec,
                "bundle_sha256": self.loaded.bundle_sha256,
                "header_rows": json.dumps(list(src.header_rows)),
            },
        )
        fr = FileResult(
            path=shown,
            sheet=grid.sheet,
            file_sha256=grid.file_sha256,
            document=document,
            rows_skipped={f"skip_rows[{i}]": 0 for i in range(len(self.c.skips))},
        )
        ref_suffix = f"@wrangle:{spec.spec}/{self.loaded.bundle_sha12}/{grid.file_sha256[:12]}"

        # Periods that don't depend on the row: a fixed cell, or each unpivoted column's label.
        fixed: tuple[Cell, tuple[date, date] | None, ParseError | None] | None = None
        if per.cell is not None:
            assert self.c.period_cell is not None
            r, c = parse_ref(per.cell.ref)
            fixed = (grid.cell(r, c), *self._period(grid.cell(r, c), self.c.period_cell))
        label_periods: dict[int, tuple[date, date] | ParseError] = {}
        for col in unpivot_cols:
            assert self.c.unpivot is not None
            try:
                label_periods[col] = self.c.unpivot(labels[col - 1])
            except ParseError as e:
                label_periods[col] = e

        entities: dict[str, RawEntity] = {}
        for r in range(src.first_data_row, len(grid.rows) + 1):
            row = grid.rows[r - 1]
            if all(c.kind == "empty" for c in row):
                continue
            fr.rows_read += 1
            if self._skipped(grid, r, skip_cols, fr):
                continue

            # Entity parts -> canonical.
            canonicals: list[str] = []
            failed = False
            for (fn, resolver), col in zip(self.c.parts, part_cols, strict=True):
                cell = grid.cell(r, col)
                if cell.kind in _READ_FAILURES:
                    self._row_reject(fr, r, cell, "entity", "read", _READ_FAILURES[cell.kind])
                    failed = True
                    break
                try:
                    cleaned = str(fn(cell.raw)) if fn else cell.raw
                except ParseError as e:
                    self._row_reject(fr, r, cell, "entity", e.op or "", e.reason)
                    failed = True
                    break
                res = resolver.resolve(cleaned)
                if res.canonical is None:
                    self._row_reject(fr, r, cell, "entity", "resolve", res.reason or "")
                    failed = True
                    break
                canonicals.append(res.canonical)
            if failed:
                continue
            entity = spec.entity.render(canonicals)
            entity_cells = tuple(f"{col_letter(c)}{r}" for c in part_cols)

            # Period for the row (column or fixed cell); unpivot is per cell, below.
            row_period: tuple[date, date] | None = None
            err: ParseError | None = None
            period_from = ""
            if period_col is not None:
                assert self.c.period_column is not None
                pcell = grid.cell(r, period_col)
                row_period, err = self._period(pcell, self.c.period_column)
                period_from = pcell.ref
            elif fixed is not None:
                pcell, row_period, err = fixed
                period_from = pcell.ref
            if err is not None:
                self._row_reject(fr, r, pcell, "period", err.op or "", err.reason)
                continue

            for m, fn, mcol in measure_cols:
                targets = [mcol] if mcol is not None else unpivot_cols
                for col in targets:
                    cell = grid.cell(r, col)
                    if cell.kind == "empty" and m.empty == "skip":
                        fr.empty_skipped += 1
                        continue
                    if row_period is not None:
                        period, pfrom = row_period, period_from
                    else:
                        lp = label_periods[col]
                        if isinstance(lp, ParseError):
                            self._cell_reject(
                                fr, cell, labels[col - 1], "period", lp.op or "", lp.reason
                            )
                            continue
                        period, pfrom = lp, labels[col - 1]
                    value = self._measure_value(fr, cell, m, fn)
                    if value is None:
                        continue
                    cell_name = f"{grid.sheet}!{cell.ref}"
                    if not self._keep(entity, m.field, period, value, cell_name):
                        continue
                    row_ = FactRow(
                        entity=entity,
                        field=m.field,
                        value=value,
                        unit=self.loaded.unit_for(m),
                        period_start=period[0],
                        period_end=period[1],
                        precision=_precision(m),  # type: ignore[arg-type]
                        source_ref=cell_name + ref_suffix,
                    )
                    fr.facts.append(
                        FactRecord(RawFact(source, row_), shown, cell_name, entity_cells, pfrom)
                    )
                    if entity not in self.crosswalk_canonicals and entity not in entities:
                        entities[entity] = RawEntity(spec.entity.kind, entity)
        fr.entities = list(entities.values())
        return fr

    def _skipped(self, grid: Grid, r: int, skip_cols, fr: FileResult) -> bool:
        for i, (kind, how) in enumerate(skip_cols):
            if kind == "blank":
                hit = all(grid.cell(r, c).kind == "empty" for c in how)
            else:
                col, rx = how
                hit = rx.search(grid.cell(r, col).raw) is not None
            if hit:
                fr.rows_skipped[f"skip_rows[{i}]"] += 1
                return True
        return False

    @staticmethod
    def _period(cell: Cell, fn: Pipeline) -> tuple[tuple[date, date] | None, ParseError | None]:
        """(period, None), or (None, the failure)."""
        if cell.kind in _READ_FAILURES:
            return None, ParseError(_READ_FAILURES[cell.kind], "read")
        try:
            return fn(cell.raw), None  # type: ignore[return-value]
        except ParseError as e:
            return None, e

    def _measure_value(
        self, fr: FileResult, cell: Cell, m: Measure, fn: Pipeline | None
    ) -> int | float | str | None:
        if cell.kind in _READ_FAILURES:
            self._cell_reject(fr, cell, cell.raw, "measure", "read", _READ_FAILURES[cell.kind])
            return None
        try:
            out = fn(cell.raw) if fn else cell.raw
        except ParseError as e:
            self._cell_reject(fr, cell, cell.raw, "measure", e.op or "", e.reason)
            return None
        if cell.kind == "empty":  # empty: reject, and the ops didn't already refuse it
            self._cell_reject(fr, cell, cell.raw, "measure", "read", "empty cell")
            return None
        if m.is_text:
            return str(out)
        assert isinstance(out, Decimal)
        return to_fact_value(out * m.scale_decimal)

    def _keep(self, entity: str, fld: str, period, value, cell_name: str) -> bool:
        key = (entity, fld, period[0], period[1])
        prior = self.seen.setdefault(key, [])
        if any(v == value for v, _ in prior):
            self.duplicates += 1
            return False
        prior.append((value, cell_name))
        return True

    def conflicts(self) -> list[dict[str, Any]]:
        return [
            {"key": [k[0], k[1], _period_label(k[2], k[3])], "cells": [c for _, c in vs]}
            for k, vs in self.seen.items()
            if len(vs) > 1
        ]


def run(
    spec: LoadedSpec | str | Path, files: Iterable[str | Path], *, force: bool = False
) -> ApplyResult:
    """The dry run: every file through the spec, nothing written anywhere.

    Raises `SpecMismatch` when a file doesn't fit the spec (header fingerprint, unless `force`;
    a missing column or sheet) and `SpecError` for a spec whose ops can't run."""
    loaded = spec if isinstance(spec, LoadedSpec) else load_project_spec(spec)
    state = _Run(loaded, force)
    results = [state.file(f) for f in files]

    crosswalk_entities = [
        RawEntity(cw.kind, canonical, cw.aliases_for(canonical))
        for cw in loaded.crosswalks.values()
        for canonical in sorted(cw.canonicals)
    ]
    counts = dict.fromkeys(STRATEGIES, 0)
    proposed: list[FuzzyMatch] = []
    for _, resolver in state.c.parts:
        for k, v in resolver.counts.items():
            counts[k] += v
        proposed.extend(resolver.fuzzy_matches)

    n_facts = sum(len(f.facts) for f in results)
    n_rejects = len(state.rejects)
    conflicts = state.conflicts()
    report = {
        "spec": loaded.spec.spec,
        "bundle_sha256": loaded.bundle_sha256,
        "approved": loaded.approved,
        "files": [
            {
                "path": f.path,
                "sheet": f.sheet,
                "file_sha256": f.file_sha256,
                "rows_read": f.rows_read,
                "rows_skipped": dict(f.rows_skipped),
                "empty_skipped": f.empty_skipped,
            }
            for f in results
        ],
        "facts": n_facts,
        "duplicates_dropped": state.duplicates,
        "rejects": {
            "total": n_rejects,
            "rate": round(reject_rate(n_facts, n_rejects), 6),
            "by_reason": dict(sorted(Counter(r["reason"] for r in state.rejects).items())),
        },
        "entities": counts,
        "conflicts_within_load": conflicts,
    }
    return ApplyResult(
        loaded=loaded,
        files=results,
        crosswalk_entities=crosswalk_entities,
        rejects=state.rejects,
        report=report,
        proposed=proposed,
        duplicates_dropped=state.duplicates,
        conflicts=conflicts,
    )


# --- gates and outputs ----------------------------------------------------------------------


def gate(
    result: ApplyResult,
    *,
    approved: bool | None = None,
    draft: bool = False,
    max_reject_rate: float = 0.02,
) -> list[str]:
    """Why apply must refuse (§8), or [] to go ahead. `approved` defaults to the spec's own
    state; `draft` waives the approval check only. (A `SpecMismatch` is raised by `run`.)"""
    reasons = []
    is_approved = result.approved if approved is None else approved
    if not draft and not is_approved:
        approval = result.loaded.spec.approval
        if approval is None:
            reasons.append(
                f"spec {result.loaded.spec.spec!r} is not approved "
                "(run `fde wrangle approve`, or load with --draft)"
            )
        else:
            reasons.append(
                f"spec {result.loaded.spec.spec!r} approval is stale: approved at bundle "
                f"{approval.bundle_sha256[:12]}, the bundle is now {result.loaded.bundle_sha12}"
            )
    rate = result.reject_rate
    if rate > max_reject_rate:
        reasons.append(
            f"reject rate {rate:.4f} ({len(result.rejects)} rejects) is over the limit "
            f"{max_reject_rate}"
        )
    return reasons


def _fact_json(rec: FactRecord) -> dict[str, Any]:
    row = rec.fact.row
    return {
        "entity": row.entity,
        "field": row.field,
        "value": row.value,
        "unit": row.unit,
        "period_start": row.period_start.isoformat() if row.period_start else None,
        "period_end": row.period_end.isoformat() if row.period_end else None,
        "precision": row.precision,
        "source_ref": row.source_ref,
        "document": rec.fact.document_source,
        "entity_cells": list(rec.entity_cells),
        "period_from": rec.period_from,
    }


def write_outputs(result: ApplyResult, out_dir: str | Path) -> dict[str, Path]:
    """report.json, rejects.jsonl, facts.preview.jsonl (first 50) and crosswalk.proposed.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out / "report.json",
        "rejects": out / "rejects.jsonl",
        "preview": out / "facts.preview.jsonl",
        "proposed": out / "crosswalk.proposed.csv",
    }
    paths["report"].write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["rejects"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.rejects),
        encoding="utf-8",
    )
    records = [r for f in result.files for r in f.facts][:PREVIEW_FACTS]
    paths["preview"].write_text(
        "".join(json.dumps(_fact_json(r), ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["crosswalk", "alias", "canonical", "score", "runner_up"])
    for p in result.proposed:
        w.writerow([p.crosswalk, p.alias, p.canonical, f"{p.score:g}", f"{p.second:g}"])
    paths["proposed"].write_text(buf.getvalue(), encoding="utf-8")
    return paths
