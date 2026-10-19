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

import hashlib
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
    tenant_id       TEXT NOT NULL DEFAULT '',
    tool_name       TEXT NOT NULL,
    content         TEXT NOT NULL,
    is_error        BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE tool_journal ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '';
"""


def _scoped_key(key: str, tenant_id: str | None) -> str:
    if not tenant_id:
        return key
    tag = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]
    return f"t:{tag}:{key}"


class ToolJournal(Protocol):
    def get(
        self, key: str, *, tenant_id: str | None = None, tool_name: str = ""
    ) -> ToolCallOutput | None: ...
    def put(
        self,
        key: str,
        result: ToolCallOutput,
        tool_name: str = "",
        *,
        tenant_id: str | None = None,
    ) -> None: ...
    def count(self) -> int: ...


# --------------------------------------------------------------------------- file


class FileJournal:
    def __init__(self, root: str | Path = ".fde/journal"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, tenant_id: str | None = None) -> Path:
        scoped = _scoped_key(key, tenant_id)
        safe = scoped.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.json"

    def get(
        self, key: str, *, tenant_id: str | None = None, tool_name: str = ""
    ) -> ToolCallOutput | None:
        p = self._path(key, tenant_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        if data.get("tenant_id", "") != (tenant_id or ""):
            return None
        recorded_tool = data.get("tool_name", "")
        if recorded_tool and tool_name and recorded_tool != tool_name:
            return None
        return ToolCallOutput(content=data["content"], is_error=data["is_error"], from_journal=True)

    def put(
        self,
        key: str,
        result: ToolCallOutput,
        tool_name: str = "",
        *,
        tenant_id: str | None = None,
    ) -> None:
        p = self._path(key, tenant_id)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        payload = {**asdict(result), "tool_name": tool_name, "tenant_id": tenant_id or ""}
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
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

    Keys are scoped by `tenant_id` so two tenants that happen to choose the same `workflow_id`
    can never hit each other's journal entries.
    """

    def __init__(self, url: str, *, min_size: int = 1, max_size: int = 8):
        from psycopg_pool import ConnectionPool

        self.url = url
        self._pool = ConnectionPool(url, min_size=min_size, max_size=max_size, open=True)
        with self._pool.connection() as conn:
            conn.execute(DDL)
            conn.commit()

    def get(
        self, key: str, *, tenant_id: str | None = None, tool_name: str = ""
    ) -> ToolCallOutput | None:
        scoped = _scoped_key(key, tenant_id)
        tid = tenant_id or ""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT content, is_error, tool_name FROM tool_journal "
                "WHERE idempotency_key = %s AND tenant_id = %s",
                (scoped, tid),
            ).fetchone()
        if row is None:
            return None
        content, is_error, recorded_tool = row
        if recorded_tool and tool_name and recorded_tool != tool_name:
            return None
        return ToolCallOutput(content=content, is_error=is_error, from_journal=True)

    def put(
        self,
        key: str,
        result: ToolCallOutput,
        tool_name: str = "",
        *,
        tenant_id: str | None = None,
    ) -> None:
        scoped = _scoped_key(key, tenant_id)
        tid = tenant_id or ""
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO tool_journal (idempotency_key, tenant_id, tool_name, content, "
                "is_error) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (idempotency_key) DO NOTHING",
                (scoped, tid, tool_name, result.content, result.is_error),
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

