"""The wrangle adapter (§7.5): an `ApplyResult` as `AdapterBatch`es for `ingest`.

The adapter doesn't parse anything itself. It loads exactly what `apply.run` (the dry run)
produced, so what a person reviewed in the dry run is what gets written.

    with store.session(tenant) as s:      # one transaction
        result, report = apply_to_tenant(s, loaded, files)
        if report is None:                # refused by a gate; nothing written
            print(gate(result, approved=result.approved, draft=False, max_reject_rate=0.02))
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ..rag.adapters.base import AdapterBatch, IngestReport, ingest
from .apply import ApplyResult, gate, run
from .spec import LoadedSpec, load_project_spec

if TYPE_CHECKING:
    from ..rag.store import TenantSession

__all__ = ["WrangleAdapter", "apply_to_tenant", "job_kind_for"]


def job_kind_for(loaded: LoadedSpec) -> str:
    """`wrangle:<spec>@<bundle_sha12>`, or `wrangle-draft:...` when the spec isn't approved at
    its current bundle hash."""
    prefix = "wrangle" if loaded.approved else "wrangle-draft"
    return f"{prefix}:{loaded.spec.spec}@{loaded.bundle_sha12}"


def _provenance(loaded: LoadedSpec, attr: str) -> str:
    """`model` / `prompt_sha256` from a spec's provenance block (written by the PR 7 agent and
    outside the approval), or "" for a hand-written spec."""
    prov = getattr(loaded.spec, "provenance", None)
    if prov is None:
        return ""
    value = prov.get(attr) if isinstance(prov, dict) else getattr(prov, attr, None)
    return str(value or "")


class WrangleAdapter:
    name = "wrangle"

    def __init__(
        self,
        spec_path: str | Path | LoadedSpec | None = None,
        files: list[Path] | None = None,
        *,
        result: ApplyResult | None = None,
        force: bool = False,
    ):
        """From a spec and files (runs `apply.run`), or from an `ApplyResult` already in hand
        (`result=`), which is what `apply_to_tenant` does so the files are parsed once."""
        if result is None:
            if spec_path is None or files is None:
                raise ValueError("WrangleAdapter needs spec_path and files, or result=")
            loaded = (
                spec_path if isinstance(spec_path, LoadedSpec) else load_project_spec(spec_path)
            )
            result = run(loaded, files, force=force)
        self.result = result
        loaded = result.loaded
        self.job_kind = job_kind_for(loaded)
        self.job_model = _provenance(loaded, "model")
        self.job_prompt_sha256 = _provenance(loaded, "prompt_sha256")

    @classmethod
    def from_result(cls, result: ApplyResult) -> WrangleAdapter:
        return cls(result=result)

    def batches(self) -> Iterator[AdapterBatch]:
        """One batch per file: the crosswalk entities (aliases from the CSV), this file's
        composite/open entities, its sheet document, its facts, and the dictionary entries
        the spec uses."""
        dictionary = self.result.loaded.dictionary_entries()
        for f in self.result.files:
            yield AdapterBatch(
                entities=[*self.result.crosswalk_entities, *f.entities],
                documents=[f.document],
                facts=[r.fact for r in f.facts],
                dictionary=list(dictionary),
            )


def apply_to_tenant(
    session: TenantSession,
    loaded: LoadedSpec | str | Path,
    files: list[Path],
    *,
    draft: bool = False,
    max_reject_rate: float = 0.02,
    force: bool = False,
) -> tuple[ApplyResult, IngestReport | None]:
    """Run the spec, apply the §8 gates, and load through `ingest(..., on_new_version=
    "replace")`. When a gate refuses, nothing is written and the report is None; the reasons
    are `gate(result, ...)`. A `SpecMismatch` propagates (also nothing written).

    Call it inside one `store.session(tenant)`, so the whole load is one transaction."""
    result = run(loaded, files, force=force)
    if gate(result, approved=result.approved, draft=draft, max_reject_rate=max_reject_rate):
        return result, None
    report = ingest(session, WrangleAdapter.from_result(result), on_new_version="replace")
    return result, report
