"""Mappers: given a world's sources and reference files, propose specs and crosswalks.

The contract a mapper keeps (the PR 7 agent will keep it too): write a wrangle project under
`project_dir`, with crosswalks named `plants` and `coa`, and return one `Proposal` per spec.
A mapper reads `sources/` and `reference/`, never `truth/` (`TruthMapper` is the one exception:
it exists to check the scorers, and must score 100%).

`HeuristicMapper` is the floor the agent has to beat (docs/wrangle-plan.md §5.3): a header
synonym dictionary, fuzzy header and name matching, and the profile's own findings. No LLM. It
writes one spec per file, so it can't see that one column holds two number formats.
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml
from rapidfuzz import fuzz

from fde_kit.wrangle.profile import ColumnProfile, Profile, profile_file
from fde_kit.wrangle.profile import profile as profile_grid
from fde_kit.wrangle.read import Grid, XlsxBook, read_grid
from fde_kit.wrangle.resolve import fuzzy_accepts, norm_key

from .generator import SOURCE_FILES, SPEC_SOURCE


@dataclass(frozen=True)
class Proposal:
    spec_path: Path
    source: str
    """Which world source ("A".."D") this spec reads."""
    file: Path


class Mapper(Protocol):
    name: str

    def propose(self, world: Path, project_dir: Path) -> list[Proposal]: ...


class TruthMapper:
    """Hands back the correct project: the scorers' self-check."""

    name = "truth"

    def propose(self, world: Path, project_dir: Path) -> list[Proposal]:
        shutil.copytree(world / "truth" / "project", project_dir, dirs_exist_ok=True)
        return [
            Proposal(
                project_dir / "specs" / f"{name}.yaml",
                source,
                world / "sources" / SOURCE_FILES[source],
            )
            for name, source in SPEC_SOURCE.items()
        ]


PLANT_LABELS = ("plant", "site", "facility", "location")
PART_LABELS = ("part", "part no", "sku", "item")
ACCOUNT_LABELS = ("acct", "account code", "gl code")
FIELD_LABELS = {
    "shipped_units": ("shipped", "qty shipped", "units shipped", "shipments"),
    "returned_units": ("returns", "returned"),
    "revenue_usd": ("revenue", "net rev", "sales"),
    "cogs_usd": ("cogs", "cost of goods"),
    "onhand_units": ("on hand", "onhand", "inventory"),
    "demand_forecast_units": ("forecast", "demand"),
    "gl_balance_usd": ("balance", "amount"),
}
COA_SYNONYMS = {
    "cogs": "cost of goods sold",
    "a/r": "accounts receivable",
    "a/p": "accounts payable",
    "cost of sales": "cost of goods sold",
    "sales": "net revenue",
}
HEADER_MIN = 90
"""A header label matches a synonym at this `token_set_ratio`."""
NAME_FUZZY_MIN = 92
COA_FUZZY_MIN = 80
_LABEL_NOISE = re.compile(r"\(.*?\)|[_/]")


def _label(text: str) -> str:
    return " ".join(_LABEL_NOISE.sub(" ", text).casefold().split())


def _label_score(label: str, synonyms: tuple[str, ...]) -> float:
    return max(fuzz.token_set_ratio(_label(label), s) for s in synonyms)


def _find_column(columns: list[ColumnProfile], synonyms: tuple[str, ...]) -> ColumnProfile | None:
    """The best text-or-number column whose label matches, first in sheet order on a tie."""
    best, best_score = None, 0.0
    for c in columns:
        score = _label_score(c.label, synonyms)
        if score >= HEADER_MIN and score > best_score:
            best, best_score = c, score
    return best


def match_plant(raw: str, canonicals: list[str]) -> str | None:
    """Exact on `norm_key`; then a `DET-07` code (city prefix + number); then a bare numeric code
    read as `..NN`, a plant number (this is the trap: irregular ERP codes end in *another*
    plant's number); then fuzzy on the name, with the margin rule."""
    key = norm_key(raw)
    by_key = {norm_key(c): c for c in canonicals}
    if key in by_key:
        return by_key[key]
    number_of = {c: int(c.rsplit(" ", 1)[1]) for c in canonicals}
    code = re.fullmatch(r"([A-Za-z]{3})-(\d+)", raw)
    if code:
        hits = [
            c
            for c in canonicals
            if c.casefold().startswith(code[1].casefold()) and number_of[c] == int(code[2])
        ]
        return hits[0] if len(hits) == 1 else None
    if raw.isdigit() and len(raw) >= 3:
        hits = [c for c in canonicals if number_of[c] == int(raw[-2:])]
        return hits[0] if len(hits) == 1 else None
    ranked = sorted(((fuzz.token_set_ratio(key, k), k) for k in by_key), reverse=True)
    if fuzzy_accepts(ranked[0][0], ranked[1][0], NAME_FUZZY_MIN):
        return by_key[ranked[0][1]]
    return None


def match_account(name: str, standard: list[str]) -> str | None:
    """A local account name to a standard account: a few synonyms, then fuzzy on the name."""
    text = COA_SYNONYMS.get(name.casefold(), name.casefold())
    names = {c.split(" ", 1)[1].casefold(): c for c in standard}
    if text in names:
        return names[text]
    scored = sorted(((fuzz.token_set_ratio(text, n), n) for n in names), reverse=True)
    if scored[0][0] >= COA_FUZZY_MIN and scored[0][0] - scored[1][0] >= 5:
        return names[scored[0][1]]
    return None


def _column_values(grid: Grid, prof: Profile, column: ColumnProfile):
    """Each cleaned value of a column's data rows, with the row number."""
    for row in range(prof.header_rows[-1] + 1, len(grid.rows) + 1):
        yield row, grid.cell(row, column.col).raw.strip()


def _number_op(column: ColumnProfile, csv_cell: bool) -> list[dict]:
    fmt = column.number_format
    number: dict = {"op": "number"}
    if fmt is not None:
        number |= {"thousands": fmt.thousands, "decimal": fmt.decimal}
        if fmt.parens_negative:
            number["parens_negative"] = True
    return [{"op": "text", "strip": True}, number] if csv_cell else [number]


def _spec_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.casefold())


class HeuristicMapper:
    name = "heuristic"

    def propose(self, world: Path, project_dir: Path) -> list[Proposal]:
        (project_dir / "specs").mkdir(parents=True, exist_ok=True)
        (project_dir / "crosswalks").mkdir(parents=True, exist_ok=True)
        shutil.copy(world / "reference" / "dictionary.yaml", project_dir / "dictionary.yaml")
        self.dictionary = yaml.safe_load((project_dir / "dictionary.yaml").read_text())
        self.plants = self._reference(world / "reference" / "plants.csv")
        self.standard = self._reference(world / "reference" / "standard_coa.csv")
        self.plant_aliases: dict[str, str] = {}
        self.account_aliases: dict[str, str] = {}
        self._sheets: dict[tuple[Path, str | None], tuple[Grid, Profile]] = {}

        proposals = []
        by_file = {name: source for source, name in SOURCE_FILES.items()}
        for path in sorted((world / "sources").iterdir()):
            book = XlsxBook(path) if path.suffix == ".xlsx" else None
            try:
                spec = self._spec_for(path, book)
            finally:
                if book is not None:
                    book.close()
            spec_path = project_dir / "specs" / f"{spec['spec']}.yaml"
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            proposals.append(Proposal(spec_path, by_file[path.name], path))

        self._write_crosswalk(project_dir / "crosswalks" / "plants.csv", self.plants, "plant")
        self._write_crosswalk(project_dir / "crosswalks" / "coa.csv", self.standard, "account")
        return proposals

    @staticmethod
    def _reference(path: Path) -> list[str]:
        with path.open(newline="", encoding="utf-8") as f:
            return [row["canonical"] for row in csv.DictReader(f)]

    def _write_crosswalk(self, path: Path, canonicals: list[str], which: str) -> None:
        aliases = self.plant_aliases if which == "plant" else self.account_aliases
        rows = [(c, c) for c in canonicals]
        rows += [(a, c) for a, c in sorted(aliases.items()) if a != c]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["alias", "canonical"])
            w.writerows(rows)

    def _sheet(self, path: Path, sheet: str | None, book: XlsxBook | None) -> tuple[Grid, Profile]:
        """Each sheet read and profiled once per run, a workbook's sheets through one open book."""
        key = (path, sheet)
        if key not in self._sheets:
            if book is not None:
                grid = read_grid(path, sheet, book=book)
                self._sheets[key] = (grid, profile_grid(grid))
            else:
                prof = profile_file(path)  # it picks the encoding; read again with that one
                grid = read_grid(path, encoding=prof.encoding) if prof.encoding else read_grid(path)
                self._sheets[key] = (grid, prof)
        return self._sheets[key]

    def _spec_for(self, path: Path, book: XlsxBook | None) -> dict:
        sheets: list[str] = list(book.workbook.sheetnames) if book is not None else []
        sheet = sheets[0] if sheets else None
        _, profile = self._sheet(path, sheet, book)
        columns = profile.columns

        crosswalks: dict = {}
        parts: list[dict] = []
        plant = _find_column(columns, PLANT_LABELS)
        if plant is not None:
            crosswalks["plants"] = {"path": "../crosswalks/plants.csv", "kind": "plant"}
            parts.append(
                {
                    "column": plant.label,
                    "ops": [{"op": "text", "strip": True}],
                    "resolve": {"crosswalk": "plants", "policy": "closed"},
                }
            )
            for name in sheets or [None]:
                for _, raw in _column_values(*self._sheet(path, name, book), plant):
                    if raw and raw not in self.plant_aliases:
                        hit = match_plant(raw, self.plants)
                        if hit is not None:
                            self.plant_aliases[raw] = hit
        kind = "plant"
        part = _find_column(columns, PART_LABELS)
        account = _find_column(columns, ACCOUNT_LABELS)
        if part is not None and part is not plant:
            kind = "plant_part"
            parts.append(
                {
                    "column": part.label,
                    "ops": [{"op": "text", "strip": True, "case": "upper"}],
                    "resolve": {"policy": "open"},
                }
            )
        elif account is not None:
            kind = "plant_account"
            crosswalks["coa"] = {"path": "../crosswalks/coa.csv", "kind": "account"}
            parts.append(
                {
                    "column": account.label,
                    "ops": [{"op": "text", "strip": True}],
                    "resolve": {"crosswalk": "coa", "policy": "closed"},
                }
            )
            self._learn_accounts(path, book, sheets, columns, account)

        source: dict = {"header_rows": list(profile.header_rows)}
        source["data_starts"] = profile.data.start
        if len(sheets) > 1:
            source["sheets"] = sheets
        elif sheets:
            source["sheet"] = sheets[0]
        if profile.skip_candidates and parts:
            source["skip_rows"] = [
                {"column": parts[0]["column"], "matches": "(?i)^(sub|grand )?total"}
            ]

        spec = {
            "spec": _spec_name(path),
            "version": 1,
            "dictionary": "../dictionary.yaml",
            "crosswalks": crosswalks,
            "source": source,
            "entity": {"kind": kind, "parts": parts},
            "period": self._period(profile),
            "measures": self._measures(profile, used={p["column"] for p in parts}, path=path),
        }
        if len(parts) > 1:
            spec["entity"]["template"] = "{0}/{1}"
        return spec

    def _learn_accounts(self, path, book, sheets, columns, account) -> None:
        name_col = next((c for c in columns if "name" in c.label.casefold()), None)
        if name_col is None:
            return
        for sheet in sheets or [None]:
            grid, prof = self._sheet(path, sheet, book)
            codes = dict(_column_values(grid, prof, account))
            names = dict(_column_values(grid, prof, name_col))
            for row, code in codes.items():
                hit = match_account(names.get(row, ""), self.standard) if code else None
                if hit is not None:
                    self.account_aliases[code] = hit

    def _period(self, profile: Profile) -> dict:
        if profile.unpivot_candidates:
            c = profile.unpivot_candidates[0]
            return {"unpivot": {"columns": {"from": c.from_, "to": c.to}, "ops": c.ops}}
        for col in profile.columns:
            if col.date_formats:
                ops = [{"op": "text", "strip": True}, {"op": "date", "format": col.date_formats[0]}]
                return {"column": {"name": col.label, "ops": ops, "grain": "month"}}
        for row in profile.title_rows:
            for cell in row.cells:
                m = re.fullmatch(r"(?i)period:\s*(.+)", cell.value)
                if m:
                    ops = [
                        {"op": "regex", "pattern": "Period: (.+)"},
                        {"op": "period", "format": "%b %Y", "grain": "month"},
                    ]
                    return {"cell": {"ref": cell.ref.split("!")[-1], "ops": ops}}
        # No period found: a spec has to name one, and this cell will not parse, so every row
        # rejects and the score shows it.
        return {"cell": {"ref": "A1", "ops": [{"op": "period", "format": "%Y", "grain": "year"}]}}

    def _measures(self, profile: Profile, used: set[str], path: Path) -> list[dict]:
        csv_cell = path.suffix != ".xlsx"
        units = {e["field"]: e["unit"] for e in self.dictionary}

        def measure(fld: str, column: ColumnProfile, column_name: str | None) -> dict:
            scale = column.scale_hint.scale if column.scale_hint else 1
            m = {"field": fld, "ops": _number_op(column, csv_cell), "unit": units[fld]}
            if column_name is not None:
                m["column"] = column_name
            if scale != 1:
                m |= {"scale": scale, "precision": scale}
            return m

        if profile.unpivot_candidates:
            first = next(c for c in profile.columns if c.label == profile.unpivot_candidates[0].to)
            fld = self._field_for(first.label)
            return [measure(fld, first, None)] if fld else []
        out, taken = [], set()
        for col in profile.columns:
            if col.label in used or col.type != "number":
                continue
            fld = self._field_for(col.label)
            if fld and fld not in taken:
                taken.add(fld)
                out.append(measure(fld, col, col.label))
        return out

    @staticmethod
    def _field_for(label: str) -> str | None:
        scored = [(_label_score(label, syns), fld) for fld, syns in FIELD_LABELS.items()]
        score, fld = max(scored)
        return fld if score >= HEADER_MIN else None
