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


def cites(result: AgentResult, value: str, _: JudgeContext) -> CheckOutcome:
    """The answer cites a Source whose `source_id`, `doc_id` or `uri` equals `value`.

    Resolved through `AgentResult.cited`: the [n] numbers in the answer, mapped back to the
    Sources tools actually returned. Prose that names the document without a number does not
    count; neither does a number that no tool produced.
    """
    hit = [s for s in result.cited if value in (s.source_id, s.doc_id, s.uri)]
    seen = [s.uri for s in result.sources]
    got = [s.uri for s in result.cited]
    return CheckOutcome(bool(hit), f"cites {value}: cited {got} of {seen}")


def cites_only(result: AgentResult, _v: Any, _: JudgeContext) -> CheckOutcome:
    """Every citation in the answer points at a Source a tool returned: nothing fabricated."""
    bad = result.fabricated_citations
    return CheckOutcome(not bad, f"cites_only: fabricated {bad}, sources {len(result.sources)}")


def cites_any(result: AgentResult, _v: Any, _: JudgeContext) -> CheckOutcome:
    """The answer cites at least one real Source. For cases where which one is not fixed."""
    return CheckOutcome(bool(result.cited), f"cites_any: cited {len(result.cited)}")


def stopped(result: AgentResult, value: str, _: JudgeContext) -> CheckOutcome:
    """The run ended for this reason: `end_turn`, `max_turns` or `replay_diverged`."""
    ok = result.stopped_reason == value
    detail = f"stopped {value}: got {result.stopped_reason!r}"
    if result.divergence is not None:
        detail += f" at turn {result.diverged_at} ({result.divergence})"
    return CheckOutcome(ok, detail)


_CONFLICT_WORDS = re.compile(
    r"\b(conflict|conflicts|conflicting|disagree|disagrees|differs|restated|revised)\b",
    re.IGNORECASE,
)


def conflict_reported(result: AgentResult, values: list[str], _: JudgeContext) -> CheckOutcome:
    """The answer names every value in `values` (thousands separators normalised like
    `contains`) and says the sources disagree: one of conflict/conflicts/conflicting/disagree/
    disagrees/differs/restated/revised as a whole word. Averaging the two numbers, or quietly
    picking one, fails."""
    hay = _norm(result.output)
    missing = [v for v in values if _norm(v) not in hay]
    word = _CONFLICT_WORDS.search(result.output)
    ok = not missing and word is not None
    found = word.group(0) if word else None
    return CheckOutcome(
        ok, f"conflict_reported {values}: missing {missing}, conflict word {found!r}"
    )


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
    "cites": cites,
    "cites_only": cites_only,
    "cites_any": cites_any,
    "stopped": stopped,
    "conflict_reported": conflict_reported,
    "judge": judge,
}
