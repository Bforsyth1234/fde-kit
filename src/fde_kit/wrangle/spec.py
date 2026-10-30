"""The mapping spec: which columns are the entity, the period and the measures, and how each
value is parsed.

A wrangle project is a directory (normally one per customer, kept in Git):

    acme/
      dictionary.yaml            # target fields, a list of DictEntry
      crosswalks/plants.csv      # header alias,canonical
      specs/planner_forecast.yaml

    loaded = load_project_spec("acme/specs/planner_forecast.yaml")
    loaded.bundle_sha12          # what approval, the job kind and every source_ref carry
    approve("acme/specs/planner_forecast.yaml", by="brooks")

Every model forbids unknown keys, so a typo is an error and not a silently ignored setting. Ops
are kept as plain dicts here; only the op *name* is checked. `wrangle/parse.py` owns what each
op's parameters mean.

The bundle hash binds the spec to the files it reads. It's sha256 over the canonical JSON
(sorted keys, no whitespace) of the spec without its `approval` block plus `{path: sha256}`
for the dictionary and each crosswalk, so editing one byte of a crosswalk after approval makes
the approval stale exactly as editing the spec would.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import re
import string
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

try:  # The store pulls in psycopg; `check` and `approve` need only the wrangle extra.
    from ..rag.store import DictEntry
except ImportError:  # pragma: no cover - exercised only without the rag/dev extras

    @dataclass
    class DictEntry:  # type: ignore[no-redef]
        """Mirror of fde_kit.rag.store.DictEntry for installs without the rag extra."""

        field: str
        definition: str
        unit: str = ""
        allowed_values: tuple[str, ...] = ()
        notes: str = ""


__all__ = [
    "OP_NAMES",
    "PERIOD_JOIN",
    "TEXT_OPS",
    "Approval",
    "Crosswalk",
    "CrosswalkRef",
    "DictEntry",
    "Entity",
    "EntityPart",
    "LoadedSpec",
    "MappingSpec",
    "Measure",
    "Period",
    "PeriodCell",
    "PeriodColumn",
    "Resolve",
    "SectionBound",
    "SheetsMatching",
    "SkipBlank",
    "SkipMatch",
    "Source",
    "SpecError",
    "Unpivot",
    "UnpivotRange",
    "approve",
    "bundle_sha256",
    "is_approved",
    "load_project_spec",
]

OP_NAMES = frozenset({"text", "regex", "map", "number", "date", "period", "constant"})
TEXT_OPS = frozenset({"text", "map", "regex", "constant"})
"""A measure whose last op is one of these produces a text fact (`value_text`)."""

SPEC_NAME = re.compile(r"^[a-z0-9_]+$")
CELL_REF = re.compile(r"^[A-Z]{1,3}[1-9][0-9]*$")
DOCUMENT_KEY_FIELDS = frozenset({"spec", "file", "path", "sheet"})
SIMILAR_WARN_DEFAULT = 90


class SpecError(ValueError):
    """The project (spec, dictionary or a crosswalk) is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _check_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, op in enumerate(ops):
        name = op.get("op")
        if name not in OP_NAMES:
            raise ValueError(f"ops[{i}]: op must be one of {sorted(OP_NAMES)}, got {name!r}")
        if name == "regex":
            pattern = op.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError(f"ops[{i}]: regex needs a string `pattern`")
            groups = _compile(pattern, f"ops[{i}].pattern").groups
            if groups != 1:
                raise ValueError(
                    f"ops[{i}]: regex pattern must have exactly one group, has {groups}"
                )
    return ops


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"{where}: invalid regex {pattern!r}: {e}") from e


Ops = Annotated[list[dict[str, Any]], AfterValidator(_check_ops), Field(default_factory=list)]


# --- source ---------------------------------------------------------------------------------


class SkipMatch(_Model):
    """Skip a row when `column`'s raw text matches the regex (`re.search`)."""

    column: str
    matches: str

    @field_validator("matches")
    @classmethod
    def _compiles(cls, v: str) -> str:
        _compile(v, "skip_rows.matches")
        return v


class SkipBlank(_Model):
    """Skip a row when every listed column is empty."""

    blank: list[str] = Field(min_length=1)


def _skip_kind(v: Any) -> str | None:
    if isinstance(v, dict):
        return "blank" if "blank" in v else "match"
    if isinstance(v, SkipBlank):
        return "blank"
    if isinstance(v, SkipMatch):
        return "match"
    return None


SkipRule = Annotated[
    Annotated[SkipMatch, Tag("match")] | Annotated[SkipBlank, Tag("blank")],
    Discriminator(_skip_kind),
]


class SectionBound(_Model):
    """A row that bounds the data: `column`'s raw text matches the regex (`re.search`), the same
    test as a `skip_rows` match rule. Used by `start_after` and `stop_before`."""

    column: str
    matches: str

    @field_validator("matches")
    @classmethod
    def _compiles(cls, v: str) -> str:
        _compile(v, "matches")
        return v


class SheetsMatching(_Model):
    """Every sheet whose name matches `matching` (`re.search`), minus `exclude` (exact names)."""

    matching: str
    exclude: list[str] = Field(default_factory=list)

    @field_validator("matching")
    @classmethod
    def _compiles(cls, v: str) -> str:
        _compile(v, "source.sheets.matching")
        return v

    @field_validator("exclude")
    @classmethod
    def _distinct(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("source.sheets.exclude lists a sheet twice")
        return v


def _sheet_names(v: list[str]) -> list[str]:
    if not v:
        raise ValueError("source.sheets list is empty")
    if len(set(v)) != len(v):
        raise ValueError("source.sheets lists a sheet twice")
    return v


Sheets = SheetsMatching | Annotated[list[str], AfterValidator(_sheet_names)]


class Source(_Model):
    sheet: str | None = None
    """XLSX: the sheet to read (this or `sheets` is required there). CSV: omit; the sheet is
    the file stem."""
    sheets: Sheets | None = None
    """XLSX, instead of `sheet`: several sheets, each loaded as its own document, in workbook
    order. `{matching: <regex>, exclude: [names]}` or a list of names. CSV: omit."""
    header_rows: list[Annotated[int, Field(ge=1)]] = Field(default_factory=lambda: [1])
    """1-based. Several rows are joined "Upper / Lower"."""
    data_starts: int | None = Field(default=None, ge=1)
    """Default: the row after the last header row. See `first_data_row`."""
    header_fingerprint: str | None = None
    document_key: str = "{spec}/{file}#{sheet}"
    """`{spec}` the spec name, `{file}` the file's base name, `{path}` the path as passed,
    `{sheet}` the sheet (CSV: the file stem)."""
    encoding: str = "utf-8-sig"
    delimiter: str | None = None
    """CSV only; None means sniffed."""
    skip_rows: list[SkipRule] = Field(default_factory=list)
    start_after: SectionBound | None = None
    """Data begins after the first row (from `first_data_row`) that matches; that row and the
    ones above it are not data. Not found -> `SpecMismatch`."""
    stop_before: SectionBound | None = None
    """Data ends before the first row after the start that matches. Not found ->
    `SpecMismatch`. Together they pick one table out of a sheet that stacks several."""

    @field_validator("header_rows")
    @classmethod
    def _headers(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("header_rows needs at least one row")
        if v != sorted(set(v)):
            raise ValueError("header_rows must be strictly increasing")
        return v

    @field_validator("header_fingerprint")
    @classmethod
    def _fingerprint(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{12}", v):
            raise ValueError("header_fingerprint is the first 12 lowercase hex chars of a sha256")
        return v

    @field_validator("document_key")
    @classmethod
    def _document_key(cls, v: str) -> str:
        try:
            names = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
        except ValueError as e:
            raise ValueError(f"document_key {v!r}: {e}") from e
        unknown = names - DOCUMENT_KEY_FIELDS
        if unknown:
            raise ValueError(
                "document_key may use only {spec}, {file}, {path} and {sheet}, "
                f"not {sorted(unknown)}"
            )
        return v

    @field_validator("encoding")
    @classmethod
    def _encoding(cls, v: str) -> str:
        try:
            codecs.lookup(v)
        except LookupError as e:
            raise ValueError(f"unknown encoding {v!r}") from e
        return v

    @field_validator("delimiter")
    @classmethod
    def _delimiter(cls, v: str | None) -> str | None:
        if v is not None and len(v) != 1:
            raise ValueError("delimiter must be a single character")
        return v

    @model_validator(mode="after")
    def _data_after_header(self) -> Source:
        if self.data_starts is not None and self.data_starts <= self.header_rows[-1]:
            raise ValueError("data_starts must be after the last header row")
        if self.sheet is not None and self.sheets is not None:
            raise ValueError("source needs at most one of `sheet` or `sheets`")
        return self

    @property
    def multi_sheet(self) -> bool:
        return self.sheets is not None

    def select_sheets(self, names: list[str]) -> list[str]:
        """The `sheets` selection from a workbook's sheet names, in workbook order. ValueError
        when it names a sheet the workbook doesn't have or selects none."""
        sel = self.sheets
        assert sel is not None
        if isinstance(sel, list):
            missing = [n for n in sel if n not in names]
            if missing:
                raise ValueError(
                    f"source.sheets names {missing}, not in the workbook; sheets: {names}"
                )
            wanted = set(sel)
            return [n for n in names if n in wanted]
        missing = [n for n in sel.exclude if n not in names]
        if missing:
            raise ValueError(
                f"source.sheets.exclude names {missing}, not in the workbook; sheets: {names}"
            )
        rx = re.compile(sel.matching)
        out = [n for n in names if rx.search(n) and n not in sel.exclude]
        if not out:
            raise ValueError(
                f"source.sheets: no sheet matches {sel.matching!r}"
                + (f" outside exclude {sel.exclude}" if sel.exclude else "")
                + f"; sheets: {names}"
            )
        return out

    @property
    def first_data_row(self) -> int:
        return self.data_starts if self.data_starts is not None else self.header_rows[-1] + 1


# --- entity ---------------------------------------------------------------------------------


class Resolve(_Model):
    crosswalk: str | None = None
    policy: Literal["closed", "open"] | None = None
    """Default: closed with a crosswalk, open without one."""
    fuzzy: bool = False
    threshold: int = Field(default=92, ge=0, le=100)
    similar_warn: int = Field(default=SIMILAR_WARN_DEFAULT, ge=0, le=100)
    """Open parts: warn about two new canonicals of this part scoring >= this (rapidfuzz
    `ratio` on a casefolded, whitespace-collapsed form). 0 turns the score warning off; pairs
    equal under `norm_key` are always reported. Warnings never merge or reject anything."""
    not_same: list[tuple[str, str]] = Field(default_factory=list)
    """Pairs a person reviewed and found to be different entities: never warned about. Exact
    strings, either order."""

    @field_validator("not_same")
    @classmethod
    def _pairs(cls, v: list[tuple[str, str]]) -> list[tuple[str, str]]:
        seen: set[frozenset[str]] = set()
        for a, b in v:
            if a == b:
                raise ValueError(f"not_same: {a!r} is paired with itself")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"not_same: the pair {a!r}, {b!r} is listed twice")
            seen.add(key)
        return v

    @model_validator(mode="after")
    def _default_policy(self) -> Resolve:
        if self.policy is None:
            self.policy = "closed" if self.crosswalk else "open"
        return self


class EntityPart(_Model):
    column: str
    ops: Ops
    resolve: Resolve = Field(default_factory=Resolve)


class Entity(_Model):
    kind: str = Field(min_length=1)
    parts: list[EntityPart] = Field(min_length=1)
    template: str = "{0}"

    @model_validator(mode="after")
    def _template_matches_parts(self) -> Entity:
        try:
            parsed = list(string.Formatter().parse(self.template))
        except ValueError as e:
            raise ValueError(f"template {self.template!r}: {e}") from e
        seen: set[str] = set()
        for _, name, spec, conv in parsed:
            if name is None:
                continue
            if spec or conv:
                raise ValueError(f"template {self.template!r}: no format specs or conversions")
            seen.add(name)
        want = {str(i) for i in range(len(self.parts))}
        if seen != want:
            raise ValueError(
                f"template {self.template!r} must use exactly the placeholders "
                f"{', '.join('{' + w + '}' for w in sorted(want, key=int))} "
                f"for {len(self.parts)} part(s)"
            )
        return self

    def render(self, canonicals: list[str]) -> str:
        return self.template.format(*canonicals)


# --- period ---------------------------------------------------------------------------------

Grain = Literal["day", "month", "quarter", "year", "instant"]


PERIOD_JOIN = "-"
"""Joins the cleaned values of a multi-column `period.column.name` before the ops run."""


class PeriodColumn(_Model):
    name: str | list[str]
    """One column label, or several whose cleaned values are joined with `PERIOD_JOIN`
    (`YEAR` 2024 and `MONTH` 1 -> `2024-1`) before the ops run."""
    ops: Ops
    grain: Grain | None = None

    @field_validator("name")
    @classmethod
    def _names(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list):
            if not v:
                raise ValueError("period.column.name list is empty")
            if len(set(v)) != len(v):
                raise ValueError("period.column.name lists a column twice")
        return v

    @property
    def names(self) -> list[str]:
        return [self.name] if isinstance(self.name, str) else list(self.name)


class PeriodCell(_Model):
    ref: str
    ops: Ops

    @field_validator("ref")
    @classmethod
    def _ref(cls, v: str) -> str:
        if not CELL_REF.match(v):
            raise ValueError(f"ref must be an A1 cell reference like B2, got {v!r}")
        return v


class UnpivotRange(_Model):
    from_: str = Field(alias="from")
    to: str


class Unpivot(_Model):
    columns: UnpivotRange | None = None
    """Header labels, inclusive, in sheet order."""
    matching: str | None = None
    """Or: every header label matching this regex (`re.search`)."""
    ops: Ops

    @model_validator(mode="after")
    def _one_selector(self) -> Unpivot:
        if (self.columns is None) == (self.matching is None):
            raise ValueError("unpivot needs exactly one of `columns` or `matching`")
        if self.matching is not None:
            _compile(self.matching, "unpivot.matching")
        return self


class Period(_Model):
    column: PeriodColumn | None = None
    cell: PeriodCell | None = None
    unpivot: Unpivot | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Period:
        given = [k for k in ("column", "cell", "unpivot") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(f"period needs exactly one of column, cell, unpivot; got {given}")
        return self


# --- measures, crosswalks, approval ---------------------------------------------------------


class Measure(_Model):
    field: str
    column: str | None = None
    """With unpivot: None (the unpivoted cells). Otherwise the column name."""
    ops: Ops
    scale: int | float = Field(default=1, gt=0)
    precision: int | float | None = Field(default=None, gt=0)
    unit: str | None = None
    empty: Literal["skip", "reject"] = "skip"
    nulls: list[str] = Field(default_factory=list)
    """Placeholders that mean "no value" in this source (`--`, `(S)`, `(NA)`): a cell whose
    stripped raw text equals one is treated as empty (so `empty` decides skip or reject) and
    counted per placeholder. No default list: what's a null in one source is data in another."""

    @field_validator("nulls")
    @classmethod
    def _nulls(cls, v: list[str]) -> list[str]:
        for token in v:
            if not token.strip():
                raise ValueError("nulls: a placeholder can't be blank (blank cells are empty)")
            if token != token.strip():
                raise ValueError(
                    f"nulls: {token!r} has surrounding whitespace; cells are compared stripped"
                )
        if len(set(v)) != len(v):
            raise ValueError("nulls: a placeholder is listed twice")
        return v

    @property
    def is_text(self) -> bool:
        """True when the value is text (`value_text`): no ops, or the last op yields a string."""
        return not self.ops or self.ops[-1]["op"] in TEXT_OPS

    @property
    def scale_decimal(self) -> Decimal:
        return Decimal(str(self.scale))

    @property
    def effective_precision(self) -> int | float | None:
        """`precision`, else `scale` when scale > 1, else None (exact)."""
        if self.precision is not None:
            return self.precision
        return self.scale if self.scale > 1 else None

    def effective_unit(self, dictionary: dict[str, DictEntry]) -> str:
        """`unit`, else the dictionary's unit for this field."""
        if self.unit is not None:
            return self.unit
        entry = dictionary.get(self.field)
        return entry.unit if entry is not None else ""

    @model_validator(mode="after")
    def _text_has_no_precision(self) -> Measure:
        if self.is_text and self.precision is not None:
            raise ValueError(f"measure {self.field!r} is text (last op), so it has no precision")
        return self


class CrosswalkRef(_Model):
    path: str
    kind: str = Field(min_length=1)


class Approval(_Model):
    by: str = Field(min_length=1)
    at: datetime
    bundle_sha256: str

    @field_validator("bundle_sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("bundle_sha256 must be 64 lowercase hex chars")
        return v


class MappingSpec(_Model):
    spec: str
    version: Literal[1] = 1
    dictionary: str
    crosswalks: dict[str, CrosswalkRef] = Field(default_factory=dict)
    source: Source = Field(default_factory=Source)
    entity: Entity
    period: Period
    measures: list[Measure] = Field(min_length=1)
    notes: str | None = None
    approval: Approval | None = None

    @field_validator("spec")
    @classmethod
    def _name(cls, v: str) -> str:
        if not SPEC_NAME.match(v):
            raise ValueError(f"spec name must match [a-z0-9_]+, got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> MappingSpec:
        unpivot = self.period.unpivot is not None
        for m in self.measures:
            if unpivot and m.column is not None:
                raise ValueError(
                    f"measure {m.field!r}: with period.unpivot the measure reads the unpivoted "
                    "cells, so `column` must be omitted"
                )
            if not unpivot and m.column is None:
                raise ValueError(f"measure {m.field!r} needs a `column` (no period.unpivot)")
        for i, part in enumerate(self.entity.parts):
            name = part.resolve.crosswalk
            if name is not None and name not in self.crosswalks:
                raise ValueError(
                    f"entity.parts[{i}].resolve.crosswalk {name!r} is not in `crosswalks` "
                    f"({sorted(self.crosswalks)})"
                )
        return self

    def hash_payload(self) -> dict[str, Any]:
        """The spec as it enters the bundle hash: JSON-mode, aliases, no approval.

        Settings added after the first release (`source.start_after`, `source.stop_before`,
        `source.sheets`, `measures[].nulls`, `resolve.similar_warn`, `resolve.not_same`) enter
        only when set (`similar_warn`: when not the default), so a spec that doesn't use them
        hashes as it did before they existed and its approval stays valid."""
        out = self.model_dump(mode="json", by_alias=True, exclude={"approval"})
        for key in _LATER_SOURCE_KEYS:
            if out["source"].get(key) is None:
                out["source"].pop(key, None)
        for m in out["measures"]:
            if not m.get("nulls"):
                m.pop("nulls", None)
        for part in out["entity"]["parts"]:
            res = part["resolve"]
            if res.get("similar_warn") == SIMILAR_WARN_DEFAULT:
                res.pop("similar_warn", None)
            if not res.get("not_same"):
                res.pop("not_same", None)
        return out


_LATER_SOURCE_KEYS = ("start_after", "stop_before", "sheets")


# --- loading --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Crosswalk:
    name: str
    kind: str
    path: Path
    mapping: dict[str, str]
    """alias -> canonical; every canonical also maps to itself."""
    canonicals: frozenset[str]
    sha256: str

    def aliases_for(self, canonical: str) -> tuple[str, ...]:
        """The CSV's aliases of `canonical` (not the canonical itself), sorted."""
        return tuple(sorted(a for a, c in self.mapping.items() if c == canonical and a != c))


@dataclass
class LoadedSpec:
    path: Path
    spec: MappingSpec
    dictionary: dict[str, DictEntry]
    crosswalks: dict[str, Crosswalk]
    files: dict[str, str]
    """{path as written in the spec: sha256 of its bytes} for the dictionary and crosswalks."""
    bundle_sha256: str

    @property
    def bundle_sha12(self) -> str:
        return self.bundle_sha256[:12]

    @property
    def approved(self) -> bool:
        return is_approved(self)

    def unit_for(self, measure: Measure) -> str:
        return measure.effective_unit(self.dictionary)

    def dictionary_entries(self) -> list[DictEntry]:
        """The dictionary entries the spec's measures use, in measure order."""
        seen: dict[str, DictEntry] = {}
        for m in self.spec.measures:
            seen.setdefault(m.field, self.dictionary[m.field])
        return list(seen.values())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256(spec: MappingSpec, files: dict[str, str]) -> str:
    payload = {"spec": spec.hash_payload(), "files": files}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256(canonical.encode("utf-8"))


def _load_dictionary(path: Path, data: bytes) -> dict[str, DictEntry]:
    try:
        raw = yaml.safe_load(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: {e}") from e
    if not isinstance(raw, list):
        raise SpecError(f"{path}: the dictionary is a YAML list of entries")
    allowed = {"field", "definition", "unit", "allowed_values", "notes"}
    out: dict[str, DictEntry] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SpecError(f"{path}[{i}]: each entry is a mapping")
        unknown = set(item) - allowed
        if unknown:
            raise SpecError(f"{path}[{i}]: unknown keys {sorted(unknown)}")
        name, definition = item.get("field"), item.get("definition")
        if not isinstance(name, str) or not name or not isinstance(definition, str):
            raise SpecError(f"{path}[{i}]: `field` and `definition` are required strings")
        if name in out:
            raise SpecError(f"{path}: field {name!r} defined twice")
        out[name] = DictEntry(
            field=name,
            definition=definition,
            unit=str(item.get("unit") or ""),
            allowed_values=tuple(str(v) for v in item.get("allowed_values") or ()),
            notes=str(item.get("notes") or ""),
        )
    return out


def _load_crosswalk(name: str, kind: str, path: Path, data: bytes) -> Crosswalk:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise SpecError(f"crosswalk {name!r} ({path}): not UTF-8: {e}") from e
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or [h.strip() for h in header] != ["alias", "canonical"]:
        raise SpecError(f"crosswalk {name!r} ({path}): header must be `alias,canonical`")
    pairs: list[tuple[int, str, str]] = []
    for lineno, row in enumerate(reader, start=2):
        if not any(c.strip() for c in row):
            continue
        if len(row) != 2:
            raise SpecError(f"crosswalk {name!r} ({path}) line {lineno}: expected 2 columns")
        alias, canonical = row[0].strip(), row[1].strip()
        if not alias or not canonical:
            raise SpecError(f"crosswalk {name!r} ({path}) line {lineno}: empty alias or canonical")
        pairs.append((lineno, alias, canonical))
    canonicals = frozenset(c for _, _, c in pairs)
    mapping = {c: c for c in canonicals}
    for lineno, alias, canonical in pairs:
        prior = mapping.get(alias)
        if prior is not None and prior != canonical:
            raise SpecError(
                f"crosswalk {name!r} ({path}) line {lineno}: {alias!r} maps to both "
                f"{prior!r} and {canonical!r}"
            )
        mapping[alias] = canonical
    return Crosswalk(name, kind, path, mapping, canonicals, _sha256(data))


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as e:
        raise SpecError(f"{what}: cannot read {path}: {e.strerror or e}") from e


def _parse_spec_text(path: Path, text: str) -> MappingSpec:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: a spec is a YAML mapping")
    return MappingSpec.model_validate(raw)


def load_project_spec(path: str | Path) -> LoadedSpec:
    """Load and validate a spec with its dictionary and crosswalks, and compute the bundle hash.

    Raises `pydantic.ValidationError` for a malformed spec and `SpecError` for problems that
    need the other files (a measure field missing from the dictionary, a bad crosswalk). Both
    are ValueErrors.
    """
    path = Path(path)
    return _load(path, _read(path, "spec").decode("utf-8-sig"))


def _load(path: Path, text: str) -> LoadedSpec:
    spec = _parse_spec_text(path, text)
    base = path.parent

    files: dict[str, str] = {}
    dict_bytes = _read(base / spec.dictionary, "dictionary")
    dictionary = _load_dictionary(base / spec.dictionary, dict_bytes)
    files[spec.dictionary] = _sha256(dict_bytes)

    crosswalks: dict[str, Crosswalk] = {}
    for name, ref in spec.crosswalks.items():
        data = _read(base / ref.path, f"crosswalk {name!r}")
        crosswalks[name] = _load_crosswalk(name, ref.kind, base / ref.path, data)
        files[ref.path] = crosswalks[name].sha256

    missing = [m.field for m in spec.measures if m.field not in dictionary]
    if missing:
        raise SpecError(f"measure field(s) {missing} not in the dictionary ({spec.dictionary})")

    return LoadedSpec(path, spec, dictionary, crosswalks, files, bundle_sha256(spec, files))


def is_approved(loaded: LoadedSpec) -> bool:
    """True when the spec carries an approval stamped at the current bundle hash."""
    approval = loaded.spec.approval
    return approval is not None and approval.bundle_sha256 == loaded.bundle_sha256


def _strip_approval(text: str) -> str:
    """Drop a top-level `approval:` block, keeping every other line exactly as written."""
    out: list[str] = []
    in_block = False
    for line in text.splitlines(keepends=True):
        if in_block:
            if not line.strip() or line[0] in " \t":
                continue
            in_block = False
        if re.match(r"approval\s*:", line):
            in_block = True
            continue
        out.append(line)
    return "".join(out).rstrip("\n") + "\n"


def approve(path: str | Path, by: str, now: datetime | None = None) -> LoadedSpec:
    """Stamp the spec's `approval` with the current bundle hash; refuse if it doesn't validate.

    The rest of the file is kept as text (comments and layout included); only the top-level
    `approval:` block is replaced, at the end of the file.
    """
    if not by.strip():
        raise ValueError("approve needs a non-empty `by`")
    path = Path(path)
    text = _strip_approval(_read(path, "spec").decode("utf-8-sig"))
    loaded = _load(path, text)  # validated without the old stamp, which may itself be stale
    at = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    text += (
        "approval:\n"
        f"  by: {json.dumps(by.strip())}\n"
        f"  at: {at.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f'  bundle_sha256: "{loaded.bundle_sha256}"\n'
    )
    path.write_text(text, encoding="utf-8")
    stamped = load_project_spec(path)
    if not is_approved(stamped):  # pragma: no cover - the hash excludes approval by design
        raise RuntimeError("approval did not round-trip; the bundle hash depends on approval")
    return stamped
