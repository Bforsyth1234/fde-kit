"""`fde runs list / tree / show` against hand-written §2 record files."""

import json

import pytest

from fde_kit import cli

PARENT = "r-aaaaaaaaaaaa"
CHILD = "r-bbbbbbbbbbbb"
PARTIAL = "r-cccccccccccc"


def _header(run_id, created_at, *, parent=None, fork_at=None, tenant=None):
    return {
        "kind": "header",
        "version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "runtime": "agent",
        "tenant_id": tenant,
        "model": "mock-model",
        "prompt_name": "default",
        "prompt_sha256": "9abf0123456789deadbeef",
        "prompt_text": "You are a careful assistant.",
        "max_turns": 10,
        "max_tokens": 2048,
        "tools": ["search_documents"],
        "parent_run_id": parent,
        "fork_at": fork_at,
        "workflow_id": None,
    }


def _body_rows():
    return [
        {"kind": "user", "turn": 1, "text": "How much is the late fee?", "actor": "user",
         "by": ""},
        {"kind": "llm", "turn": 1, "actor": "model", "text": "", "stop_reason": "tool_use",
         "tool_calls": [{"id": "toolu_01", "name": "search_documents",
                         "input": {"query": "late fee"}}],
         "raw_content": [], "usage": {"input_tokens": 812, "output_tokens": 41}},
        {"kind": "tool", "turn": 1, "tool_use_id": "toolu_01", "name": "search_documents",
         "input": {"query": "late fee"}, "content": "[#1] Lease 101", "is_error": False,
         "sources": [{"source_id": "s1", "uri": "lease://101", "title": "Lease 101",
                      "doc_id": "d1", "content_hash": "abc", "tenant_id": "summit"}],
         "approval": None},
        {"kind": "llm", "turn": 2, "actor": "model", "text": "The late fee is $50 [1].",
         "stop_reason": "end_turn", "tool_calls": [], "raw_content": [],
         "usage": {"input_tokens": 900, "output_tokens": 12}},
    ]


def _result_row():
    return {"kind": "result", "output": "The late fee is $50 [1].",
            "stopped_reason": "end_turn", "turns": 2, "total_ms": 2412,
            "sources": [{"source_id": "s1"}], "cited": ["s1"], "fabricated_citations": []}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    # parent: complete, oldest
    _write(tmp_path / f"{PARENT}.jsonl",
           [_header(PARENT, "2026-10-22T10:00:00Z", tenant="summit"), *_body_rows(),
            _result_row()])
    # fork child of the parent, newest
    child_rows = _body_rows()
    child_rows.insert(3, {"kind": "user", "turn": 2, "text": "Answer in one word.",
                          "actor": "user", "by": "cli"})
    _write(tmp_path / f"{CHILD}.jsonl",
           [_header(CHILD, "2026-10-22T12:00:00Z", parent=PARENT, fork_at=1), *child_rows,
            {**_result_row(), "output": "$50 [1].", "stopped_reason": "end_turn", "turns": 2}])
    # run that never finished: no result row
    _write(tmp_path / f"{PARTIAL}.jsonl",
           [_header(PARTIAL, "2026-10-22T11:00:00Z"), *_body_rows()[:3]])
    # unreadable: not a header first
    (tmp_path / "garbage.jsonl").write_text('{"kind":"user","turn":1}\nnot json\n')
    return tmp_path


def _run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_list_orders_newest_first(runs_dir, capsys):
    code, out = _run(capsys, "runs", "list", "--dir", str(runs_dir))
    assert code == 0
    ids = [line.split("│")[1].strip() for line in out.splitlines() if line.startswith("│ r-")]
    assert ids == [CHILD, PARTIAL, PARENT]


def test_list_columns_for_a_complete_run(runs_dir, capsys):
    _, out = _run(capsys, "runs", "list", "--dir", str(runs_dir))
    parent_line = next(line for line in out.splitlines() if line.startswith(f"│ {PARENT}"))
    for cell in ("2026-10-22T10:00:00Z", "agent", "summit", "mock-model",
                 "default@9abf01234567", "end_turn"):
        assert cell in parent_line
    assert "│ 2 " in parent_line


def test_list_marks_fork_parent(runs_dir, capsys):
    _, out = _run(capsys, "runs", "list", "--dir", str(runs_dir))
    child_line = next(line for line in out.splitlines() if line.strip().startswith(f"│ {CHILD}"))
    assert f"{PARENT}@1" in child_line


def test_list_shows_incomplete_and_unreadable(runs_dir, capsys):
    _, out = _run(capsys, "runs", "list", "--dir", str(runs_dir))
    partial_line = next(line for line in out.splitlines() if PARTIAL in line)
    assert "incomplete" in partial_line
    garbage_line = next(line for line in out.splitlines() if "garbage" in line)
    assert "unreadable" in garbage_line


def test_list_empty_dir(tmp_path, capsys):
    code, out = _run(capsys, "runs", "list", "--dir", str(tmp_path))
    assert code == 0
    assert f"no runs in {tmp_path}" in out


def test_list_missing_dir(tmp_path, capsys):
    missing = tmp_path / "nope"
    code, out = _run(capsys, "runs", "list", "--dir", str(missing))
    assert code == 0
    assert "no runs in" in out


def test_list_uses_env_dir(runs_dir, monkeypatch, capsys):
    monkeypatch.setenv("FDE_RUNS_DIR", str(runs_dir))
    code, out = _run(capsys, "runs", "list")
    assert code == 0 and CHILD in out


def test_tree_from_child_shows_parent_above_and_marks_requested(runs_dir, capsys):
    code, out = _run(capsys, "runs", "tree", CHILD, "--dir", str(runs_dir))
    assert code == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0].startswith(PARENT)
    assert lines[1].startswith("  " + CHILD)
    assert "<-- this run" in lines[1] and "<-- this run" not in lines[0]
    assert f"fork of {PARENT} at turn 1" in lines[1]
    assert PARTIAL not in out


def test_tree_from_parent_shows_forks_below(runs_dir, capsys):
    code, out = _run(capsys, "runs", "tree", PARENT, "--dir", str(runs_dir))
    assert code == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0].startswith(PARENT) and "<-- this run" in lines[0]
    assert lines[1].startswith("  " + CHILD)


def test_tree_accepts_filename_stem(runs_dir, capsys):
    (runs_dir / "renamed.jsonl").write_text(
        json.dumps(_header("r-dddddddddddd", "2026-10-22T13:00:00Z")) + "\n")
    code, out = _run(capsys, "runs", "tree", "renamed", "--dir", str(runs_dir))
    assert code == 0 and "r-dddddddddddd" in out


def test_tree_unknown_id_exits_1(runs_dir, capsys):
    code, out = _run(capsys, "runs", "tree", "r-000000000000", "--dir", str(runs_dir))
    assert code == 1
    assert "unknown run" in out


def test_show_prints_header_and_every_row_kind(runs_dir, capsys):
    code, out = _run(capsys, "runs", "show", CHILD, "--dir", str(runs_dir))
    assert code == 0
    assert f"run_id: {CHILD}" in out
    assert f"parent_run_id: {PARENT}" in out
    assert "fork_at: 1" in out
    assert "You are a careful assistant." not in out  # prompt_text omitted
    assert "user   user: How much is the late fee?" in out
    assert "user   user: Answer in one word." in out
    assert "llm    stop_reason=tool_use  tools: search_documents" in out
    assert "tool   search_documents  is_error=False  sources=1" in out
    assert "llm    stop_reason=end_turn  tools: -" in out
    assert "result stopped=end_turn  cited=1  $50 [1]." in out


def test_show_unknown_id_exits_1(runs_dir, capsys):
    code, out = _run(capsys, "runs", "show", "r-000000000000", "--dir", str(runs_dir))
    assert code == 1 and "unknown run" in out


def test_loader_rejects_bad_files(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(cli.RunFileError, match="empty"):
        cli._load_run_file(empty)
    no_header = tmp_path / "nh.jsonl"
    no_header.write_text('{"kind":"user","turn":1}\n')
    with pytest.raises(cli.RunFileError, match="header"):
        cli._load_run_file(no_header)
    bad_version = tmp_path / "v.jsonl"
    bad_version.write_text(json.dumps({**_header("r-1", "x"), "version": 2}) + "\n")
    with pytest.raises(cli.RunFileError, match="version"):
        cli._load_run_file(bad_version)
    bad_json = tmp_path / "j.jsonl"
    bad_json.write_text(json.dumps(_header("r-1", "x")) + "\n{oops\n")
    with pytest.raises(cli.RunFileError, match="not JSON"):
        cli._load_run_file(bad_json)


def test_loader_skips_blank_lines(runs_dir):
    header, rows = cli._load_run_file(runs_dir / f"{PARENT}.jsonl")
    assert header["run_id"] == PARENT
    assert [r["kind"] for r in rows] == ["user", "llm", "tool", "llm", "result"]


def test_show_full_prints_tool_inputs_and_outputs(runs_dir, capsys):
    code, out = _run(capsys, "runs", "show", CHILD, "--dir", str(runs_dir), "--full")
    assert code == 0
    assert 'call search_documents: {"query": "late fee"}' in out  # the model's tool call
    assert 'input: {"query": "late fee"}' in out  # the tool row's input
    assert "output: [#1] Lease 101" in out  # brackets survive: no rich markup
    # Without --full, none of that detail is printed.
    _, short = _run(capsys, "runs", "show", CHILD, "--dir", str(runs_dir))
    assert "output: [#1] Lease 101" not in short


def test_show_full_prints_sql_and_errors_and_clips_long_output(tmp_path, capsys):
    rows = [
        {"kind": "header", "version": 1, "run_id": "r-sqlsqlsqlsql",
         "created_at": "2026-10-23T00:00:00Z",
         "runtime": "agent", "tenant_id": "pharma-a", "model": "m", "prompt_name": "default",
         "prompt_sha256": "0" * 64, "parent_run_id": None, "fork_at": None},
        {"kind": "tool", "turn": 1, "name": "query_table", "input": {"sql": "SELECT 1"},
         "content": "Query rejected: one statement only", "is_error": True, "sources": []},
        {"kind": "tool", "turn": 2, "name": "query_table", "input": {"sql": "SELECT f.fact_id"},
         "content": "x" * 5000, "is_error": False, "sources": []},
    ]
    (tmp_path / "r-sqlsqlsqlsql.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    code, out = _run(capsys, "runs", "show", "r-sqlsqlsqlsql", "--dir", str(tmp_path), "--full")
    assert code == 0
    assert "sql: SELECT 1" in out and "error: Query rejected: one statement only" in out
    assert "sql: SELECT f.fact_id" in out and "[3800 more chars]" in out
