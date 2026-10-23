"""`FactRow` with `precision` and `source_ref`, whether or not the store has them yet.

PR 4 workstream 1-A adds both fields to `store.FactRow`; until that lands this module provides a
subclass that carries them, so adapters can be written against the final shape now.

# TODO(integrator): drop shim, use store.FactRow
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Any

from ..store import FactRow as _StoreFactRow

HAS_NATIVE_FIELDS = {"precision", "source_ref"} <= {f.name for f in fields(_StoreFactRow)}

if HAS_NATIVE_FIELDS:  # pragma: no cover - depends on which store is installed
    FactRow = _StoreFactRow
else:

    @dataclass
    class FactRow(_StoreFactRow):  # type: ignore[no-redef]
        """store.FactRow plus the two PR 4 columns (shim)."""

        precision: int | None = None
        source_ref: str = ""


def make_fact_row(
    *,
    entity: str,
    field: str,
    value: float | int | str,
    unit: str = "",
    period_start: date | None = None,
    period_end: date | None = None,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    precision: int | None = None,
    source_ref: str = "",
) -> Any:
    """A FactRow (a `store.FactRow` instance either way) with precision and source_ref set."""
    return FactRow(
        entity=entity,
        field=field,
        value=value,
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        doc_id=doc_id,
        chunk_id=chunk_id,
        precision=precision,
        source_ref=source_ref,
    )
