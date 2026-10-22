"""TEMPORARY: input dataclasses for the facts table, copied from the PR 3 spec (§3).

Lets `demo_corpus` build `FactRow` / `DictEntry` values before `store.py` grows them. Once the
store lands, delete this module and import the real classes from `.store` instead; the shapes
here are identical to the spec so the swap is a one-line change at the import site.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class DictEntry:
    field: str
    definition: str
    unit: str = ""
    allowed_values: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class FactRow:
    """INPUT shape for `TenantSession.add_facts`."""

    entity: str
    """Canonical name or alias; resolved to an entity_id by the store."""
    field: str
    value: float | int | str
    """A number becomes value_num, a string value_text."""
    unit: str = ""
    period_start: date | None = None
    period_end: date | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
