"""Idempotency journal for tool executions.

The failure this guards against: a tool runs to completion, then the worker dies before Temporal
records the activity as complete. Temporal will retry the activity on another worker. Without a
journal, the tool runs twice (an email is sent twice, a ticket is created twice). With it, the
retry finds the recorded result under the same key and returns it without re-executing.

The key is derived in the workflow from (workflow_id, turn, tool_use_id), which is stable across
retries and across worker restarts.

Two backends:

- FileJournal: a directory of JSON files with atomic renames. Fine for one worker on one box.
  It is NOT shared across machines, which is the whole point of the retry scenario, so it is
  the development default and nothing more.
- PostgresJournal: one table, one upsert. Any worker anywhere sees any other worker's record.
  This is the one to deploy.

`journal_from_url` picks by scheme so the worker needs one setting (`FDE_JOURNAL_URL`).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .models import ToolCallOutput

DDL = """
CREATE TABLE IF NOT EXISTS tool_journal (
    idempotency_key TEXT PRIMARY KEY,
    tool_name       TEXT NOT NULL,
    content         TEXT NOT NULL,
    is_error        BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class ToolJournal(Protocol):
    def get(self, key: str) -> ToolCallOutput | None: ...
    def put(self, key: str, result: ToolCallOutput, tool_name: str = "") -> None: ...
    def count(self) -> int: ...


# --------------------------------------------------------------------------- file


class FileJournal:
    def __init__(self, root: str | Path = ".fde/journal"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.json"

    def get(self, key: str) -> ToolCallOutput | None:
        p = self._path(key)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return ToolCallOutput(content=data["content"], is_error=data["is_error"], from_journal=True)

    def put(self, key: str, result: ToolCallOutput, tool_name: str = "") -> None:
        p = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(result), f)
        os.replace(tmp, p)  # atomic on POSIX; a crash leaves either the old file or the new one

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.json"))

    def __repr__(self) -> str:
        return f"FileJournal({self.root})"


# --------------------------------------------------------------------------- postgres


class PostgresJournal:
    """Idempotency journal in one Postgres table.

    `put` is an INSERT ... ON CONFLICT DO NOTHING. If two workers somehow race to record the
    same key (one slow worker finishing an attempt Temporal already timed out), the first write
    wins and the second is a no-op; both return the same content either way.
    """

    def __init__(self, url: str, *, min_size: int = 1, max_size: int = 8):
        from psycopg_pool import ConnectionPool

        self.url = url
        self._pool = ConnectionPool(url, min_size=min_size, max_size=max_size, open=True)
        with self._pool.connection() as conn:
            conn.execute(DDL)
            conn.commit()

    def get(self, key: str) -> ToolCallOutput | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT content, is_error FROM tool_journal WHERE idempotency_key = %s", (key,)
            ).fetchone()
        if row is None:
            return None
        return ToolCallOutput(content=row[0], is_error=row[1], from_journal=True)

    def put(self, key: str, result: ToolCallOutput, tool_name: str = "") -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO tool_journal (idempotency_key, tool_name, content, is_error) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (idempotency_key) DO NOTHING",
                (key, tool_name, result.content, result.is_error),
            )
            conn.commit()

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM tool_journal").fetchone()[0]

    def close(self) -> None:
        self._pool.close()

    def __repr__(self) -> str:
        return "PostgresJournal(<url>)"


# --------------------------------------------------------------------------- factory


def journal_from_url(url: str | None) -> ToolJournal:
    """`postgresql://...` -> PostgresJournal; `file:<dir>` or a bare path -> FileJournal."""
    if not url:
        return FileJournal()
    if url.startswith(("postgresql://", "postgres://")):
        return PostgresJournal(url)
    if url.startswith("file:"):
        return FileJournal(url[len("file:") :])
    return FileJournal(url)
