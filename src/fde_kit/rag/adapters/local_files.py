"""A directory a tenant supplies: markdown, text and HTML files as documents.

Optional `facts.jsonl` at the root adds facts, one JSON object per line:

    {"document": "leases/a.md", "entity": "Unit 4B", "field": "rent", "value": 1800,
     "unit": "USD", "period_start": "2026-01-01", "period_end": null}

`precision` and `source_ref` are honoured when present. Every entity a fact names (and the
`entity` argument) is upserted as kind "entity".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from ..store import FactRow
from .base import AdapterBatch, RawDocument, RawEntity, RawFact
from .edgar import html_to_text

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
HTML_SUFFIXES = {".htm", ".html"}
FACTS_FILE = "facts.jsonl"


def _date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


class LocalFilesAdapter:
    name = "local_files"

    def __init__(self, root: Path, *, glob: str = "**/*", entity: str | None = None):
        self.root = Path(root)
        self.glob = glob
        self.entity = entity
        self.skipped: list[str] = []
        """Relative paths skipped (PDFs and other unsupported files), filled by batches()."""

    def _documents(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        self.skipped = []
        for path in sorted(self.root.glob(self.glob)):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if rel == FACTS_FILE or any(part.startswith(".") for part in Path(rel).parts):
                continue
            suffix = path.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
            elif suffix in HTML_SUFFIXES:
                text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
            elif suffix == ".pdf":
                log.warning("local_files: skipping %s (PDF extraction is not supported)", rel)
                self.skipped.append(rel)
                continue
            else:
                self.skipped.append(rel)
                continue
            docs.append(
                RawDocument(
                    source=rel,
                    title=path.name,
                    text=text,
                    metadata={"path": rel, "suffix": suffix},
                )
            )
        return docs

    def _facts(self) -> list[RawFact]:
        path = self.root / FACTS_FILE
        if not path.exists():
            return []
        out: list[RawFact] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                entity = d.get("entity") or self.entity
                if not entity:
                    raise ValueError("no entity (set it on the line or pass entity=)")
                row = FactRow(
                    entity=entity,
                    field=d["field"],
                    value=d["value"],
                    unit=d.get("unit", ""),
                    period_start=_date(d.get("period_start")),
                    period_end=_date(d.get("period_end")),
                    precision=d.get("precision"),
                    source_ref=d.get("source_ref", ""),
                )
                out.append(RawFact(d["document"], row))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{n}: bad fact line: {e}") from None
        return out

    def batches(self) -> Iterator[AdapterBatch]:
        documents = self._documents()
        facts = self._facts()
        names = [self.entity] if self.entity else []
        names += [f.row.entity for f in facts]
        entities = [RawEntity("entity", n) for n in dict.fromkeys(names)]
        yield AdapterBatch(entities=entities, documents=documents, facts=facts)
