"""Record, replay and fork of a run. Offline: MockProvider and ToolResult-returning tools."""

import json

import pytest

from fde_kit import (
    Agent,
    ReplayDiverged,
    ReplayProvider,
    ReplayRegistry,
    RunRecord,
    RunRecorder,
    Settings,
    Source,
    ToolRegistry,
    ToolResult,
    fork_session,
    live_llm_session,
    new_run_id,
    replay_session,
    write_record_from_result,
)
from fde_kit.llm import MockProvider
from fde_kit.record import default_runs_dir


def src(i: int, tenant: str | None = None) -> Source:
    return Source(
        source_id=f"chunk-{i}",
        uri=f"docs/{i}.md",
        title=f"Doc {i}",
        content_hash="ab",
        tenant_id=tenant,
    )


def make_registry(tenant_id: str | None = None) -> ToolRegistry:
    reg = ToolRegistry(tenant_id=tenant_id)
    calls: list[tuple[str, dict]] = []
    reg.calls = calls  # type: ignore[attr-defined]

    @reg.tool
    def search(query: str) -> ToolResult:
        """Search."""
        calls.append(("search", {"query": query}))
        return ToolResult(
            "[#1] Doc 1\nLate fee is $50.\n\n[#2] Doc 2\nRent is due on the 1st.",
            sources=(src(1, tenant_id), src(2, tenant_id)),
        )

    @reg.tool
    def search_more(query: str) -> ToolResult:
        """Search again; one already-seen source and one new one."""
        calls.append(("search_more", {"query": query}))
        return ToolResult(
            "[#1] again\n\n[#2] Doc 3\nPets allowed.",
            sources=(src(2, tenant_id), src(3, tenant_id)),
        )

    @reg.tool
    def plain(x: int) -> str:
        """Doubles."""
        calls.append(("plain", {"x": x}))
        return str(x * 2)

    return reg


SCRIPT = [
    {"text": "Looking.", "tool_calls": [{"name": "search", "input": {"query": "fee"}}]},
    {"tool_calls": [{"name": "search_more", "input": {"query": "pets"}}]},
    "The late fee is $50 [1] and pets are allowed [3].",
]

SETTINGS = Settings(provider="mock", otel_enabled=False, system_prompt="Be careful.")


def agent(reg=None, script=SCRIPT, **kw) -> Agent:
    return Agent(reg or make_registry(), provider=MockProvider(script), settings=SETTINGS, **kw)


def record_run(tmp_path, reg=None, script=SCRIPT, prompt="How much is the late fee?"):
    path = tmp_path / "parent.jsonl"
    result = agent(reg, script).run(prompt, record=path)
    return path, result


def lines(path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------------ the file


def test_recorded_file_has_header_first_rows_in_order_result_last(tmp_path):
    path, result = record_run(tmp_path)
    rows = lines(path)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["header", "user", "llm", "tool", "llm", "tool", "llm", "result"]
    # `kind` is the first key on every line.
    assert all(ln.startswith('{"kind": "') for ln in path.read_text().splitlines())

    h = rows[0]
    assert h["version"] == 1 and h["runtime"] == "agent" and h["run_id"] == result.run_id
    assert h["run_id"].startswith("r-") and len(h["run_id"]) == 14
    assert h["tenant_id"] is None and h["model"] == "mock"
    assert h["prompt_name"] == "inline" and h["prompt_text"] == "Be careful."
    assert h["prompt_sha256"] == result.prompt_sha256
    assert h["max_turns"] == 10 and h["max_tokens"] == 2048
    assert h["tools"] == ["search", "search_more", "plain"]
    assert h["parent_run_id"] is None and h["fork_at"] is None and h["workflow_id"] is None
    assert h["created_at"].endswith("Z")

    assert rows[1] == {
        "kind": "user",
        "turn": 1,
        "text": "How much is the late fee?",
        "actor": "user",
        "by": "",
    }
    llm1 = rows[2]
    assert llm1["turn"] == 1 and llm1["actor"] == "model" and llm1["text"] == "Looking."
    assert llm1["stop_reason"] == "tool_use"
    assert llm1["tool_calls"] == [
        {"id": "toolu_mock_0_0", "name": "search", "input": {"query": "fee"}}
    ]
    assert llm1["raw_content"][1]["type"] == "tool_use"
    assert llm1["usage"] == {"input_tokens": 10, "output_tokens": 5}

    tool1 = rows[3]
    assert tool1["turn"] == 1 and tool1["name"] == "search" and tool1["is_error"] is False
    assert tool1["tool_use_id"] == "toolu_mock_0_0" and tool1["input"] == {"query": "fee"}
    assert tool1["approval"] is None
    # Raw content: local [#i] markers, not the ledger's run-wide numbers.
    assert tool1["content"].startswith("[#1] Doc 1")
    assert tool1["sources"] == [src(1).to_dict(), src(2).to_dict()]
    # Second tool row also raw: the loop would have shown the model "[2] again".
    assert rows[5]["content"].startswith("[#1] again")

    res = rows[-1]
    assert res["output"] == result.output and res["stopped_reason"] == "end_turn"
    assert res["turns"] == 3 and isinstance(res["total_ms"], int)
    assert [s["source_id"] for s in res["sources"]] == ["chunk-1", "chunk-2", "chunk-3"]
    assert res["cited"] == ["chunk-1", "chunk-3"] and res["fabricated_citations"] == []
    assert result.record_path == str(path)


def test_run_record_load_and_accessors(tmp_path):
    path, result = record_run(tmp_path)
    rec = RunRecord.load(path)
    assert rec.run_id == result.run_id
    assert len(rec.rows) == 7
    assert len(rec.llm_rows()) == 3 and len(rec.tool_rows()) == 2 and len(rec.user_rows()) == 1
    assert rec.result["output"] == result.output


def test_load_rejects_bad_version_and_missing_header(tmp_path):
    path, _ = record_run(tmp_path)
    rows = lines(path)

    bad = tmp_path / "bad_version.jsonl"
    rows[0]["version"] = 2
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="version"):
        RunRecord.load(bad)

    no_header = tmp_path / "no_header.jsonl"
    no_header.write_text("\n".join(json.dumps(r) for r in rows[1:]) + "\n")
    with pytest.raises(ValueError, match="header"):
        RunRecord.load(no_header)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        RunRecord.load(empty)


def test_recorder_writes_error_result_row_on_exception(tmp_path):
    path = tmp_path / "boom.jsonl"
    header = {"run_id": new_run_id(), "model": "mock"}
    with pytest.raises(RuntimeError):
        with RunRecorder(path, header) as rec:
            rec.write({"kind": "user", "turn": 1, "text": "q", "actor": "user", "by": ""})
            rec.write({"kind": "llm", "turn": 1, "actor": "model", "text": ""})
            raise RuntimeError("provider exploded")
    rows = lines(path)
    assert rows[0]["kind"] == "header" and rows[0]["version"] == 1
    assert rows[-1]["kind"] == "result"
    assert rows[-1]["stopped_reason"] == "error" and rows[-1]["error"] == "RuntimeError"
    assert rows[-1]["turns"] == 1
    assert RunRecord.load(path).result["error_message"] == "provider exploded"


def test_agent_run_records_error_row_when_provider_raises(tmp_path):
    class Boom:
        model = "mock"

        def complete(self, **kw):
            raise ValueError("no")

    path = tmp_path / "agent_boom.jsonl"
    ag = Agent(make_registry(), provider=Boom(), settings=SETTINGS)
    with pytest.raises(ValueError):
        ag.run("q", record=path)
    rows = lines(path)
    assert [r["kind"] for r in rows] == ["header", "user", "result"]
    assert rows[-1]["stopped_reason"] == "error" and rows[-1]["error"] == "ValueError"


def test_tenant_id_carried_in_header_and_sources(tmp_path):
    path, _ = record_run(tmp_path, reg=make_registry(tenant_id="summit"))
    rec = RunRecord.load(path)
    assert rec.header["tenant_id"] == "summit"
    assert all(s["tenant_id"] == "summit" for row in rec.tool_rows() for s in row["sources"])
    assert all(s["tenant_id"] == "summit" for s in rec.result["sources"])


def test_write_record_from_result_matches_streamed_file(tmp_path):
    path, result = record_run(tmp_path)
    header = RunRecord.load(path).header
    other = write_record_from_result(tmp_path / "rebuilt.jsonl", header, result)
    assert other.read_text() == path.read_text()


def test_new_run_id_and_default_runs_dir():
    a, b = new_run_id(), new_run_id()
    assert a != b and a.startswith("r-") and len(a) == 14
    assert str(default_runs_dir(Settings(provider="mock"))) == "runs"
    assert str(default_runs_dir(Settings(provider="mock", runs_dir="/tmp/x"))) == "/tmp/x"
    assert Settings(provider="mock").record_runs is False


# ------------------------------------------------------------------ replay


def test_exact_replay_is_byte_identical_and_calls_no_tools(tmp_path):
    path, parent = record_run(tmp_path)
    rec = RunRecord.load(path)

    base = make_registry()
    session = replay_session(rec, base)
    # The replay agent has a provider that would blow up if used, and tools that count calls.
    child_path = tmp_path / "child.jsonl"
    replayed = Agent(base, provider=MockProvider(["WRONG"]), settings=SETTINGS).run(
        "How much is the late fee?", session=session, record=child_path
    )

    assert replayed.output == parent.output
    assert replayed.stopped_reason == "end_turn" and replayed.turns == parent.turns
    assert [s.source_id for s in replayed.cited] == [s.source_id for s in parent.cited]
    assert replayed.sources == parent.sources
    assert base.calls == []  # nothing re-computed
    # The model-facing content was re-derived through the ledger: run-wide numbers again.
    assert replayed.steps[1].tool_results[0]["content"].startswith("[2] again")
    assert replayed.steps[1].tool_results[0]["citations"] == [2, 3]

    # Recording a replay produces a file whose rows (ignoring header identity) match the parent.
    def body(p):
        rows = lines(p)
        return [r for r in rows[1:-1]], {k: v for k, v in rows[-1].items() if k != "total_ms"}

    assert body(child_path) == body(path)
    child_header = lines(child_path)[0]
    assert child_header["parent_run_id"] == parent.run_id and child_header["fork_at"] is None
    assert child_header["run_id"] != parent.run_id


def test_replay_registry_delegates_schemas_to_base_and_carries_tenant(tmp_path):
    path, _ = record_run(tmp_path, reg=make_registry(tenant_id="summit"))
    base = make_registry(tenant_id="summit")
    reg = ReplayRegistry(RunRecord.load(path), base)
    assert reg.names() == base.names()
    assert reg.api_schemas() == base.api_schemas()
    assert reg.get("search") is base.get("search")
    assert reg.requires_approval("search") is False
    assert reg.tenant_id == "summit"
    assert isinstance(reg, ToolRegistry)


def test_replay_provider_is_indexed_by_assistant_count(tmp_path):
    path, _ = record_run(tmp_path)
    prov = ReplayProvider(RunRecord.load(path))
    assert prov.model == "mock"
    msgs = [{"role": "user", "content": "q"}]
    r1 = prov.complete(system="s", messages=msgs, tools=[], max_tokens=10)
    assert r1.tool_calls[0].name == "search" and r1.stop_reason == "tool_use"
    # An injected user message between turns does not shift the index.
    msgs += [
        {"role": "assistant", "content": r1.raw_content},
        {"role": "user", "content": "tool results"},
        {"role": "user", "content": "injected instruction"},
    ]
    r2 = prov.complete(system="s", messages=msgs, tools=[], max_tokens=10)
    assert r2.tool_calls[0].name == "search_more"
    msgs += [{"role": "assistant", "content": r2.raw_content}]
    r3 = prov.complete(system="s", messages=msgs, tools=[], max_tokens=10)
    assert r3.text == SCRIPT[2] and r3.tool_calls == []
    assert r3.usage.input_tokens == 10
    # Past the end of the record with no live provider: divergence, not a silent answer.
    msgs += [{"role": "assistant", "content": r3.raw_content}]
    with pytest.raises(ReplayDiverged):
        prov.complete(system="s", messages=msgs, tools=[], max_tokens=10)


def test_divergence_sets_stopped_reason_and_details(tmp_path):
    path, parent = record_run(tmp_path)
    rows = lines(path)
    # Edit the second tool row's input so the replayed model's request no longer matches.
    second_tool = [r for r in rows if r["kind"] == "tool"][1]
    second_tool["input"] = {"query": "parking"}
    edited = tmp_path / "edited.jsonl"
    edited.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    base = make_registry()
    child = tmp_path / "diverged.jsonl"
    result = agent(base).run(
        "How much is the late fee?",
        session=replay_session(RunRecord.load(edited), base),
        record=child,
    )
    assert result.stopped_reason == "replay_diverged"
    assert result.diverged_at == 2
    assert result.divergence == {
        "expected": {"name": "search_more", "input": {"query": "parking"}},
        "got": {"name": "search_more", "input": {"query": "pets"}},
    }
    assert result.turns == 1  # the diverging turn is not kept as a step
    assert base.calls == []
    res_row = lines(child)[-1]
    assert res_row["stopped_reason"] == "replay_diverged" and res_row["diverged_at"] == 2
    assert "expected search_more" in str(
        ReplayDiverged(2, result.divergence["expected"], result.divergence["got"])
    )


def test_divergence_on_wrong_tool_name_and_exhausted_record(tmp_path):
    path, _ = record_run(tmp_path)
    reg = ReplayRegistry(RunRecord.load(path), make_registry())
    reg.set_turn(1)
    with pytest.raises(ReplayDiverged) as ei:
        reg.execute("plain", {"query": "fee"})
    assert ei.value.turn == 1 and ei.value.expected["name"] == "search"
    assert ei.value.got == {"name": "plain", "input": {"query": "fee"}}
    reg.execute("search", {"query": "fee"})
    reg.set_turn(2)
    reg.execute("search_more", {"query": "pets"})
    with pytest.raises(ReplayDiverged) as ei:
        reg.execute("search", {"query": "x"})
    assert ei.value.expected is None and ei.value.turn == 2


# ------------------------------------------------------------------ live llm


def test_live_llm_replay_serves_tools_from_record_and_calls_live_provider(tmp_path):
    path, parent = record_run(tmp_path)
    base = make_registry()
    live = MockProvider(
        [
            {"tool_calls": [{"name": "search", "input": {"query": "fee"}}]},
            {"tool_calls": [{"name": "search_more", "input": {"query": "pets"}}]},
            "Reworded: pets OK [3], fee $50 [1].",
        ]
    )
    session = live_llm_session(RunRecord.load(path), base, live)
    result = agent(base).run("How much is the late fee?", session=session)

    assert len(live.calls) == 3  # every LLM call went to the live provider
    assert live.calls[0]["system"] == "Be careful."
    assert base.calls == []  # tools were served from the record
    assert result.output == "Reworded: pets OK [3], fee $50 [1]."
    # Evidence path identical to the parent: same sources, same cited set.
    assert result.sources == parent.sources
    assert {s.source_id for s in result.cited} == {s.source_id for s in parent.cited}
    assert result.model == "mock"


def test_live_llm_replay_diverges_when_live_model_asks_for_something_else(tmp_path):
    path, _ = record_run(tmp_path)
    base = make_registry()
    live = MockProvider([{"tool_calls": [{"name": "plain", "input": {"x": 1}}]}, "done"])
    result = agent(base).run("q", session=live_llm_session(RunRecord.load(path), base, live))
    assert result.stopped_reason == "replay_diverged" and result.diverged_at == 1
    assert result.divergence["got"] == {"name": "plain", "input": {"x": 1}}
    assert result.divergence["expected"] == {"name": "search", "input": {"query": "fee"}}


# ------------------------------------------------------------------ fork


def test_fork_replays_through_at_then_injects_and_continues_live(tmp_path):
    path, parent = record_run(tmp_path)
    base = make_registry()
    live = MockProvider(
        [
            "unused-0",
            "unused-1",
            # index 2: the first live call (two assistant messages already in the conversation)
            {"tool_calls": [{"name": "plain", "input": {"x": 21}}]},
            "Forked answer: 42 and the fee is $50 [1].",
        ]
    )
    session = fork_session(RunRecord.load(path), base, live, at=2, instruction="Now double 21.")
    child = tmp_path / "child.jsonl"
    result = agent(base).run("How much is the late fee?", session=session, record=child)

    assert result.output == "Forked answer: 42 and the fee is $50 [1]."
    assert result.stopped_reason == "end_turn" and result.turns == 4
    # Turns 1-2 came from the record (no real tool ran); turn 3's tool ran live.
    assert base.calls == [("plain", {"x": 21})]
    assert len(live.calls) == 2
    # The injected instruction sits right before the first live LLM call, after the tool results.
    first_live_msgs = live.calls[0]["messages"]
    assert first_live_msgs[-1] == {"role": "user", "content": "Now double 21."}
    assert first_live_msgs[-2]["role"] == "user"  # tool results of turn 2
    assert first_live_msgs[-2]["content"][0]["type"] == "tool_result"
    assert sum(1 for m in first_live_msgs if m["role"] == "assistant") == 2
    # Evidence from the parent is still numbered the same way in the fork.
    assert [s.source_id for s in result.sources] == ["chunk-1", "chunk-2", "chunk-3"]
    assert [s.source_id for s in result.cited] == ["chunk-1"]

    rows = lines(child)
    h = rows[0]
    assert h["parent_run_id"] == parent.run_id and h["fork_at"] == 2
    assert h["run_id"] != parent.run_id
    assert [r["kind"] for r in rows] == [
        "header",
        "user",
        "llm",
        "tool",
        "llm",
        "tool",
        "user",
        "llm",
        "tool",
        "llm",
        "result",
    ]
    users = [r for r in rows if r["kind"] == "user"]
    assert users[0]["text"] == "How much is the late fee?" and users[0]["turn"] == 1
    assert users[1]["text"] == "Now double 21." and users[1]["turn"] == 3
    assert rows[8]["name"] == "plain" and rows[8]["content"] == "42" and rows[8]["turn"] == 3
    # Parent untouched.
    assert RunRecord.load(path).run_id == parent.run_id
    assert len(lines(path)) == 8
    # The child replays exactly on its own.
    base2 = make_registry()
    again = agent(base2).run(
        "How much is the late fee?", session=replay_session(RunRecord.load(child), base2)
    )
    assert again.output == result.output and base2.calls == []


def test_fork_at_zero_injects_before_first_turn(tmp_path):
    path, parent = record_run(tmp_path)
    base = make_registry()
    live = MockProvider(["Fresh start."])
    session = fork_session(RunRecord.load(path), base, live, at=0, instruction="Ignore docs.")
    result = agent(base).run("q", session=session)
    assert result.output == "Fresh start." and result.turns == 1
    assert [m["role"] for m in live.calls[0]["messages"]] == ["user", "user"]
    assert live.calls[0]["messages"][1]["content"] == "Ignore docs."
    assert session.fork_at == 0 and session.parent_run_id == parent.run_id


# ------------------------------------------------------------------ unchanged behaviour


def test_run_without_session_or_record_is_unchanged():
    result = agent().run("q")
    assert result.run_id == "" and result.record_path == ""
    assert result.diverged_at is None and result.divergence is None
    assert result.stopped_reason == "end_turn"
    assert result.steps[0].tool_results[0]["content"].startswith("[1] Doc 1")


def test_agent_level_recorder_is_used_by_run(tmp_path):
    ag = agent()
    path = tmp_path / "via_recorder.jsonl"
    with RunRecorder(path, ag.build_header("q")) as rec:
        ag.recorder = rec
        result = ag.run("q")
    assert result.run_id == rec.run_id
    assert [r["kind"] for r in lines(path)][-1] == "result"
    assert lines(path)[-1]["stopped_reason"] == "end_turn"
