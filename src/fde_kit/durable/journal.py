"""Idempotency journal for tool executions.

The failure this guards against: a tool runs to completion, then the worker dies before Temporal
records the activity as complete. Temporal will retry the activity on another worker. Without a
journal, the tool runs twice (an email is sent twice, a ticket is created twice). With it, the
retry finds the recorded result under the same key and returns it without re-executing.

The key is derived in the workflow from (workflow_id, turn, tool_use_id), which is stable across
retries and across worker restarts.

This v0 is a directory of JSON files with atomic writes. It is deliberately simple: the point of
the project is to demonstrate the pattern, and the storage backend is a one-class swap (Redis,
DynamoDB, Postgres) that doesn't change the workflow or the activity.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import ToolCallOutput


class ToolJournal:
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

    def put(self, key: str, result: ToolCallOutput) -> None:
        p = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(result), f)
        os.replace(tmp, p)  # atomic on POSIX; a crash leaves either the old file or the new one

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.json"))
