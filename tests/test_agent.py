from fde_kit.agent import Agent
from fde_kit.builtin_tools import registry
from fde_kit.config import Settings
from fde_kit.llm import MockProvider


def make_agent(script, **overrides):
    settings = Settings(provider="mock", otel_enabled=False, **overrides)
    return Agent(registry, provider=MockProvider(script), settings=settings)


def test_single_tool_round_trip():
    agent = make_agent(
        [
            {"tool_calls": [{"name": "calculator", "input": {"expression": "17 * 23"}}]},
            "17 * 23 = 391.",
        ]
    )
    result = agent.run("What is 17 * 23?")
    assert result.output == "17 * 23 = 391."
    assert result.tools_called == ["calculator"]
    assert result.turns == 2
    assert result.stopped_reason == "end_turn"
    assert result.steps[0].tool_results[0]["content"] == "391"
    assert result.usage.input_tokens == 20 and result.usage.output_tokens == 10


def test_tool_result_message_shape_matches_api_contract():
    provider = MockProvider(
        [{"tool_calls": [{"name": "current_date", "input": {}}]}, "done"]
    )
    settings = Settings(provider="mock", otel_enabled=False)
    Agent(registry, provider=provider, settings=settings).run("date?")

    # Second LLM call must contain: user prompt, assistant tool_use, user tool_result (first).
    msgs = provider.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    tool_use = msgs[1]["content"][0]
    tool_result = msgs[2]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == tool_use["id"]
    assert "is_error" not in tool_result


def test_tool_error_is_returned_to_model_not_raised():
    agent = make_agent(
        [
            {"tool_calls": [{"name": "calculator", "input": {"expression": "1 /"}}]},
            "That expression was invalid.",
        ]
    )
    result = agent.run("broken math")
    assert result.steps[0].tool_results[0]["is_error"] is True
    assert result.output == "That expression was invalid."


def test_parallel_tool_calls_in_one_turn():
    agent = make_agent(
        [
            {
                "tool_calls": [
                    {"name": "lookup_policy", "input": {"topic": "warranty"}},
                    {"name": "lookup_policy", "input": {"topic": "sla"}},
                ]
            },
            "Both looked up.",
        ]
    )
    result = agent.run("warranty and sla?")
    assert result.tools_called == ["lookup_policy", "lookup_policy"]
    assert len(result.steps[0].tool_results) == 2


def test_max_turns_terminates_runaway_loop():
    endless = [{"tool_calls": [{"name": "current_date", "input": {}}]}] * 50
    agent = make_agent(endless, max_turns=3)
    result = agent.run("loop forever")
    assert result.turns == 3
    assert result.stopped_reason == "max_turns"


def test_tools_schema_is_sent_every_call():
    provider = MockProvider(["hi"])
    settings = Settings(provider="mock", otel_enabled=False)
    Agent(registry, provider=provider, settings=settings).run("hello")
    names = {t["name"] for t in provider.calls[0]["tools"]}
    assert names == {"calculator", "current_date", "lookup_policy"}
