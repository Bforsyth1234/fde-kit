"""The query_table guard: a pure keyword filter, tested without Postgres.

It accepts one SELECT (optionally with a CTE) and rejects everything else it can recognise.
It is not the isolation boundary; tests/test_rag_facts.py covers transaction_read_only and RLS.
"""

from __future__ import annotations

import pytest

from fde_kit.rag import SqlRejected, check_select
from fde_kit.rag.sqlguard import FORBIDDEN, strip_comments

# ------------------------------------------------------------------ accepts


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select field, value_num from rag_facts where unit = 'USD'",
        "  \n SELECT * FROM rag_facts f JOIN rag_entities e ON e.entity_id = f.entity_id",
        "WITH late AS (SELECT * FROM rag_facts WHERE field = 'late_fee') SELECT count(*) FROM late",
        "with a as (select 1 as x), b as (select x + 1 as y from a) select y from b",
        "SELECT * FROM rag_facts WHERE entity_id IN (SELECT entity_id FROM rag_entities)",
        "SELECT (SELECT max(value_num) FROM rag_facts) AS top",
        "SELECT field FROM rag_facts ORDER BY field LIMIT 5 OFFSET 2",
        "SELECT value_text FROM rag_facts WHERE value_text LIKE '%rent%'",
        "SELECT 1 UNION ALL SELECT 2",
    ],
)
def test_accepts_single_select(sql: str):
    assert check_select(sql).lower().lstrip().startswith(("select", "with"))


def test_trailing_semicolon_is_stripped():
    assert check_select("SELECT 1;") == "SELECT 1"
    assert check_select("SELECT 1 ;  \n") == "SELECT 1"


def test_comments_are_stripped_before_checking():
    cleaned = check_select("-- a note\nSELECT 1 /* inline */ AS x -- trailing")
    assert "--" not in cleaned and "/*" not in cleaned
    assert cleaned.split() == ["SELECT", "1", "AS", "x"]


def test_forbidden_word_inside_a_comment_does_not_reject():
    assert check_select("SELECT 1 -- then DELETE everything").strip() == "SELECT 1"
    assert check_select("/* update: nothing */ SELECT 1").strip() == "SELECT 1"


def test_strip_comments_handles_multiline_block():
    assert strip_comments("SELECT /* a\nb\nc */ 1").split() == ["SELECT", "1"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT deleted_at FROM rag_facts",
        "SELECT reset_count, updated_by, settings, insertion, showcase FROM rag_facts",
        "SELECT dropped, altered, created_at, granted, locked, analyzed FROM rag_facts",
        "SELECT info FROM rag_facts WHERE field = 'security_deposit_months'",
        "SELECT * FROM rag_facts WHERE value_text = 'undone'",
    ],
)
def test_forbidden_tokens_as_substrings_of_identifiers_are_fine(sql: str):
    check_select(sql)


def test_guard_is_not_literal_aware_and_errs_strict():
    """A forbidden word inside a string literal is still rejected. Documented, not fixed: the
    filter is a keyword filter and being too strict is the safe side."""
    _reason("SELECT * FROM rag_facts WHERE value_text = 'do not call'")


# ------------------------------------------------------------------ rejects


def _reason(sql: str) -> str:
    with pytest.raises(SqlRejected) as exc:
        check_select(sql)
    assert isinstance(exc.value, ValueError)
    assert exc.value.reason and str(exc.value) == exc.value.reason
    return exc.value.reason


@pytest.mark.parametrize("word", FORBIDDEN)
def test_rejects_each_forbidden_token_as_a_whole_word(word: str):
    assert word in _reason(f"SELECT 1 FROM rag_facts WHERE {word} = 1")
    assert word in _reason(f"SELECT 1 FROM rag_facts WHERE {word.upper()} = 1")


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE rag_facts SET value_num = 0",
        "DELETE FROM rag_facts",
        "INSERT INTO rag_facts (tenant_id) VALUES ('x')",
        "TRUNCATE rag_facts",
        "DROP TABLE rag_facts",
        "COPY rag_facts TO '/tmp/out'",
        "SET app.tenant_id = 'harbor'",
        "SHOW app.tenant_id",
        "EXPLAIN SELECT 1",
        "VALUES (1)",
        "TABLE rag_facts",
        "",
        "   ",
    ],
)
def test_rejects_anything_that_is_not_a_select(sql: str):
    _reason(sql)


def test_rejects_two_statements():
    assert ";" in _reason("SELECT 1; DELETE FROM rag_facts")
    assert ";" in _reason("SELECT 1; SELECT 2")
    assert ";" in _reason("SELECT 1;; ")


def test_rejects_cte_that_writes():
    reason = _reason("WITH x AS (DELETE FROM rag_facts RETURNING *) SELECT * FROM x")
    assert "delete" in reason


def test_rejects_dollar_quoting():
    assert "dollar" in _reason("SELECT $$ DELETE FROM rag_facts $$")


def test_rejects_catalog_and_system_access():
    assert "pg_" in _reason("SELECT relname FROM pg_class")
    assert "pg_" in _reason("SELECT pg_sleep(10)")
    assert "pg_" in _reason("SELECT PG_READ_FILE('/etc/passwd')")
    assert "information_schema" in _reason("SELECT * FROM information_schema.tables")


def test_rejects_setting_functions():
    assert "set_config" in _reason("SELECT set_config('app.tenant_id', 'summit', true)")
    assert "current_setting" in _reason("SELECT current_setting('app.tenant_id')")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * INTO temp t FROM rag_facts",
        "SELECT * INTO TEMPORARY t FROM rag_facts",
        "SELECT * INTO UNLOGGED t FROM rag_facts",
        "SELECT * INTO t FROM rag_facts",
    ],
)
def test_rejects_select_into(sql: str):
    assert "into" in _reason(sql)


def test_rejects_security_definer_and_do_blocks():
    _reason("SELECT 1 FROM rag_facts WHERE security = 1")
    _reason("DO $$ BEGIN END $$")


def test_non_string_is_rejected_not_crashed():
    _reason(None)  # type: ignore[arg-type]
