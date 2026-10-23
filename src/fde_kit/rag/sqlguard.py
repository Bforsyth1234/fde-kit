"""The guard in front of `TenantSession.query`: one read-only SELECT, nothing else.

This is a keyword filter and says so. It exists to turn the obvious mistakes and the obvious
attacks (`; DELETE`, `COPY ... TO`, `SELECT ... INTO`, `pg_sleep`, `set_config`) into a short
rejection before anything reaches Postgres. It is not the boundary. The boundary is that the
statement runs under `SET LOCAL transaction_read_only = on`, under a statement timeout, as the
app role, inside a tenant session whose forced RLS policy filters every table it can name.
A statement that slips past this filter still cannot write, and still cannot see another
tenant's rows. tests/test_sqlguard.py covers the filter; tests/test_rag_facts.py covers the
boundary.
"""

from __future__ import annotations

import re

__all__ = ["SqlRejected", "check_select"]


class SqlRejected(ValueError):
    """query_table refused the statement before sending it: not a single SELECT, or it
    touches something off-limits. `.reason` is a short human string."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SELECT = re.compile(r"^\s*(with\b.*?)?select\b", re.IGNORECASE | re.DOTALL)

# Whole words only: `deleted_at` and `reset_count` are column names, not statements. `pg_` is a
# prefix (pg_class, pg_sleep, pg_read_file, ...); `into` is only a problem when it is the
# SELECT ... INTO form that creates a table.
FORBIDDEN = (
    "insert", "update", "delete", "merge", "truncate", "drop", "alter", "create", "grant",
    "revoke", "copy", "vacuum", "analyze", "set", "reset", "show", "listen", "notify", "do",
    "call", "execute", "prepare", "lock", "security", "information_schema", "current_setting",
    "set_config",
)  # fmt: skip
_FORBIDDEN = re.compile(
    r"\b(?:" + "|".join(FORBIDDEN) + r")\b"
    r"|\bpg_"
    r"|\binto\s+(?:temp|temporary|unlogged)?\s*\w",
    re.IGNORECASE,
)


def strip_comments(sql: str) -> str:
    """Remove `-- ...` and `/* ... */` comments. Not string-literal aware, on purpose: a
    comment marker inside a literal is removed too, which can only make the filter stricter."""
    return _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql))


def check_select(sql: str) -> str:
    """Return the cleaned statement (comments and one trailing `;` removed) or raise
    `SqlRejected`."""
    if not isinstance(sql, str):
        raise SqlRejected("sql must be a string")
    cleaned = strip_comments(sql).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        raise SqlRejected("empty statement")
    if ";" in cleaned:
        raise SqlRejected("one statement only (found ';')")
    if "$$" in cleaned:
        raise SqlRejected("dollar-quoted blocks are not allowed")
    if not _SELECT.match(cleaned):
        raise SqlRejected("only a single SELECT (optionally with a CTE) is allowed")
    hit = _FORBIDDEN.search(cleaned)
    if hit:
        word = hit.group(0).split()[0].lower()
        raise SqlRejected(f"{word!r} is not allowed in a read-only query")
    return cleaned
