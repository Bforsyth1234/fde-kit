"""Check implementations for the eval harness.

Each check receives the AgentResult and returns (passed, detail). A case passes only if every
check passes. Deterministic checks come first; `judge` is the only one that calls a model, and it
is skipped (counted as pass with a note) when no judge provider is configured, so offline runs
still exercise the rest of the set.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fde_kit.agent import AgentResult
from fde_kit.llm import LLMProvider


@dataclass
class CheckOutcome:
    passed: bool
    detail: str


Scorer = Callable[[AgentResult, Any, "JudgeContext"], CheckOutcome]


@dataclass
class JudgeContext:
    provider: LLMProvider | None
    prompt: str


_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, and drop thousands separators so '65,536' == '65536'."""
    return " ".join(_THOUSANDS.sub("", s).lower().split())


def exact(result: AgentResult, value: str, _: JudgeContext) -> CheckOutcome:
    ok = result.output.strip() == value
    return CheckOutcome(ok, f"exact {value!r}: got {result.output.strip()!r}")


def contains(result: AgentResult, value: str, _: JudgeContext) -> CheckOutcome:
    ok = _norm(value) in _norm(result.output)
    return CheckOutcome(ok, f"contains {value!r}")


def any_of(result: AgentResult, values: list[str], _: JudgeContext) -> CheckOutcome:
    hay = _norm(result.output)
    hit = next((v for v in values if _norm(v) in hay), None)
    return CheckOutcome(hit is not None, f"any_of {values}: matched {hit!r}")


def regex(result: AgentResult, pattern: str, _: JudgeContext) -> CheckOutcome:
    ok = re.search(pattern, result.output) is not None
    return CheckOutcome(ok, f"regex /{pattern}/")


def tool_called(result: AgentResult, name: str, _: JudgeContext) -> CheckOutcome:
    ok = name in result.tools_called
    return CheckOutcome(ok, f"tool_called {name}: called {result.tools_called}")


def no_tool(result: AgentResult, _v: Any, _: JudgeContext) -> CheckOutcome:
    ok = not result.tools_called
    return CheckOutcome(ok, f"no_tool: called {result.tools_called}")


_JUDGE_SYSTEM = (
    "You are a strict grader. You will be given a task prompt, a model's answer, and a rubric. "
    "Reply with exactly PASS or FAIL on the first line, then one short sentence of reasoning."
)


def judge(result: AgentResult, rubric: str, ctx: JudgeContext) -> CheckOutcome:
    if ctx.provider is None:
        return CheckOutcome(True, "judge: skipped (no judge provider)")
    user = (
        f"TASK PROMPT:\n{ctx.prompt}\n\nMODEL ANSWER:\n{result.output}\n\n"
        f"RUBRIC:\n{rubric}\n\nDoes the answer satisfy the rubric?"
    )
    resp = ctx.provider.complete(
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        tools=[],
        max_tokens=200,
    )
    first = resp.text.strip().splitlines()[0].strip().upper() if resp.text.strip() else ""
    ok = first.startswith("PASS")
    return CheckOutcome(ok, f"judge: {resp.text.strip()[:200]}")


SCORERS: dict[str, Scorer] = {
    "exact": exact,
    "contains": contains,
    "any_of": any_of,
    "regex": regex,
    "tool_called": tool_called,
    "no_tool": no_tool,
    "judge": judge,
}
