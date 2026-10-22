"""`fde run` record / replay / fork flags, driven through `cli.main` with the mock provider."""

import json
from pathlib import Path

import pytest

from fde_kit import Agent, Settings, cli
from fde_kit.builtin_tools import registry
from fde_kit.llm import MockProvider
from fde_kit.record import RunRecord

# The cite-01 golden script: one real builtin tool call, then a cited answer.
SCRIPT = [
    {"tool_calls": [{"name": "lookup_policy", "input": {"topic": "sla"}}]},
    "The uptime SLA is 99.9% monthly, measured on the API gateway [1].",
]
PROMPT = "What is the uptime SLA? Cite your source."


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("FDE_PROVIDER", "mock")
    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    monkeypatch.setenv("FDE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("FDE_RECORD_RUNS", raising=False)
    monkeypatch.setenv("FDE_SYSTEM_PROMPT", "Be careful and cite.")
    return tmp_path


@pytest.fixture
def parent(env) -> Path:
    """A real record of the cite-01 run over the builtin registry."""
    settings = Settings(provider="mock", otel_enabled=False, system_prompt="Be careful and cite.")
    agent = Agent(registry, provider=MockProvider(SCRIPT), settings=settings)
    path = env / "parent.jsonl"
    result = agent.run(PROMPT, record=path)
    assert result.stopped_reason == "end_turn"
    assert "99.9" in result.output
    return path


def _run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- replay


def test_replay_is_identical(parent, capsys):
    code, out = _run(capsys, "run", "--replay", str(parent))
    assert code == 0
    assert out.count("99.9% monthly") == 2  # recorded, then replayed
    assert "identical" in out
    assert "differs" not in out


def test_replay_prompt_comes_from_record(parent, capsys):
    code, out = _run(capsys, "run", "--replay", str(parent), "ignored prompt")
    assert code == 0
    assert "identical" in out


def test_replay_of_edited_record_diverges(parent, capsys):
    rows = _rows(parent)
    for row in rows:
        if row["kind"] == "tool":
            row["input"] = {"topic": "refund policy"}
    edited = parent.with_name("edited.jsonl")
    edited.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    code, out = _run(capsys, "run", "--replay", str(edited))
    assert code == 3
    assert "replay diverged at turn 1" in out
    assert 'expected lookup_policy({"topic": "refund policy"})' in out
    assert 'got lookup_policy({"topic": "sla"})' in out


def test_replay_live_llm_runs_without_real_tools(parent, capsys, monkeypatch):
    from fde_kit import builtin_tools

    def boom(topic: str):
        raise AssertionError("real tool called during --live-llm replay")

    monkeypatch.setattr(builtin_tools.registry.get("lookup_policy"), "fn", boom)
    code, out = _run(capsys, "run", "--replay", str(parent), "--live-llm")
    assert code == 0
    assert "99.9% monthly" in out  # the recorded output
    assert "[mock: script exhausted]" in out  # the live (empty-script) mock
    assert "differs" in out
    assert "live llm" in out


# --------------------------------------------------------------------------- fork


def test_fork_writes_child_with_lineage(parent, env, capsys):
    code, out = _run(capsys, "run", "--fork", str(parent), "--at", "1", "Now check refunds.")
    assert code == 0
    parent_id = RunRecord.load(parent).run_id
    children = list((env / "runs").glob("*.jsonl"))
    assert len(children) == 1
    child = RunRecord.load(children[0])
    assert child.header["parent_run_id"] == parent_id
    assert child.header["fork_at"] == 1
    assert children[0].stem == child.run_id
    assert f"forked {parent_id} at turn 1 -> {children[0]}" in out
    # turn 1 replayed from the record, the instruction injected before turn 2
    assert [r["name"] for r in child.tool_rows()] == ["lookup_policy"]
    assert [r["text"] for r in child.user_rows()] == [PROMPT, "Now check refunds."]
    assert child.result["stopped_reason"] == "end_turn"


# --------------------------------------------------------------------------- record


def test_record_runs_setting_saves_to_runs_dir(env, capsys, monkeypatch):
    monkeypatch.setenv("FDE_RECORD_RUNS", "1")
    code, out = _run(capsys, "run", "hello")
    assert code == 0
    files = list((env / "runs").glob("*.jsonl"))
    assert len(files) == 1
    rec = RunRecord.load(files[0])
    assert files[0].stem == rec.run_id
    assert f"run_id {rec.run_id} saved to {files[0]}" in out
    assert rec.header["parent_run_id"] is None
    assert rec.result is not None


def test_run_without_record_runs_writes_nothing(env, capsys):
    code, out = _run(capsys, "run", "hello")
    assert code == 0
    assert not (env / "runs").exists()
    assert "saved to" not in out


def test_record_flag_writes_path(env, capsys):
    target = env / "custom" / "r.jsonl"
    code, out = _run(capsys, "run", "--record", str(target), "hello")
    assert code == 0
    rec = RunRecord.load(target)
    assert rec.header["prompt_text"] == "Be careful and cite."
    assert f"run_id {rec.run_id} saved to {target}" in out


# --------------------------------------------------------------------------- argparse errors


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--record", "x.jsonl", "--replay", "y.jsonl"],
        ["run", "--record", "x.jsonl", "--fork", "y.jsonl", "--at", "1", "go"],
        ["run", "--replay", "x.jsonl", "--fork", "y.jsonl", "--at", "1", "go"],
        ["run", "--fork", "y.jsonl", "go"],  # missing --at
        ["run", "--fork", "y.jsonl", "--at", "1"],  # missing instruction
        ["run", "--replay"],  # no PATH
        ["run", "--at", "1", "go"],  # --at without --fork
        ["run"],  # no prompt
    ],
)
def test_exclusive_flag_combinations_are_usage_errors(env, argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_replay_missing_file_is_an_error(env, capsys):
    code, out = _run(capsys, "run", "--replay", str(env / "nope.jsonl"))
    assert code == 2
