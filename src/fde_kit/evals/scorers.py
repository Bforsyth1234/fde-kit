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


# --- value-aware checks -----------------------------------------------------------------------
#
# A live model writes the same number many ways: "$51,750 million", "$51.75 billion",
# "USD 51,750,000,000". String checks written against one phrasing fail the others, so these
# checks parse every amount in the answer into a plain number and compare values instead.

_SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}
# A number, optional decimals, optional scale. Single-letter scales (M, B, K) only count right
# after a currency sign ("$4.1B"), so "Unit 3B" is not three billion.
_AMOUNT = re.compile(
    r"(?P<cur>\$|usd\s*)?"
    r"(?<![\w.])(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s*(?P<word>thousand|million|billion|trillion|mn|bn)\b"
    r"|(?P<letter>[kmb])\b)?",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[\d+\]")
DEFAULT_TOL = 0.0025  # relative: "$16.07 billion" matches 16,071.7 million; 41,172 != 40,905


def amounts(text: str) -> list[float]:
    """Every amount in `text` as a plain number, with scale words applied. Citation markers
    like [3] are dropped first so they are not read as numbers."""
    out: list[float] = []
    for m in _AMOUNT.finditer(_CITATION.sub(" ", text)):
        n = float(m["num"].replace(",", "") + (m["dec"] or ""))
        if m["word"]:
            n *= _SCALES[m["word"].lower()]
        elif m["letter"] and m["cur"]:
            n *= _SCALES[m["letter"].lower()]
        out.append(n)
    return out


def _target(value: Any) -> tuple[float, float, str]:
    """A check value is "$51,750 million", a bare number, or {"value": ..., "tol": ...}."""
    tol = DEFAULT_TOL
    if isinstance(value, dict):
        tol = float(value.get("tol", tol))
        value = value["value"]
    if isinstance(value, (int, float)):
        return float(value), tol, str(value)
    parsed = amounts(str(value))
    if len(parsed) != 1:
        raise ValueError(f"amount check value {value!r} must hold exactly one number")
    return parsed[0], tol, str(value)


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def _has_amount(text: str, value: Any) -> bool:
    want, tol, _ = _target(value)
    return any(_near(got, want, tol) for got in amounts(text))


def amount(result: AgentResult, value: Any, _: JudgeContext) -> CheckOutcome:
    """The answer states this amount, in any format: "$16,071.7 million", "$16.07 billion" and
    "16,071,700,000" all match 16,071.7 million (relative tolerance 0.25% by default)."""
    _, _, label = _target(value)
    return CheckOutcome(_has_amount(result.output, value), f"amount {label}")


def no_amount(result: AgentResult, value: Any, _: JudgeContext) -> CheckOutcome:
    """The answer does NOT state this amount in any format. For leak checks: another tenant's
    number must not appear, whether written as 81,288 million or $81.3 billion. Uses a wider
    default tolerance (1%) than `amount`, so it errs toward flagging a leak."""
    if not isinstance(value, dict):
        value = {"value": value, "tol": 0.01}
    _, _, label = _target(value)
    leaked = _has_amount(result.output, value)
    return CheckOutcome(not leaked, f"no_amount {label}" + (": found" if leaked else ""))


_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS = [
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    re.compile(
        rf"\b(?P<mon>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b", re.I
    ),
    re.compile(
        rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{_MONTH})\.?,?\s+(?P<y>\d{{4}})\b", re.I
    ),
]


def dates(text: str) -> set[str]:
    """Every calendar date in `text` as ISO YYYY-MM-DD: "2026-12-01", "December 1, 2026",
    "Dec. 1st, 2026" and "1 December 2026" are all 2026-12-01."""
    out: set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            month = int(m["m"]) if "m" in m.groupdict() and m["m"] else _MONTHS[m["mon"].lower()]
            out.add(f"{int(m['y']):04d}-{month:02d}-{int(m['d']):02d}")
    return out


def date(result: AgentResult, value: str, _: JudgeContext) -> CheckOutcome:
    """The answer states this date (ISO in the check), in ISO or written-out form."""
    found = dates(result.output)
    return CheckOutcome(value in found, f"date {value}: found {sorted(found)}")


def cites_one_of(result: AgentResult, values: list[str], _: JudgeContext) -> CheckOutcome:
    """The answer cites at least one of these sources. For facts several documents state
    identically (every lease has the same grace-period clause)."""
    hit = [s.uri for s in result.cited if {s.source_id, s.doc_id, s.uri} & set(values)]
    return CheckOutcome(bool(hit), f"cites_one_of: cited {hit or [s.uri for s in result.cited]}")


_CONFLICT_WORDS = re.compile(
    r"\b(conflict\w*|disagree\w*|differ\w*|restat\w*|revis\w*|inconsisten\w*|discrepan\w*)",
    re.IGNORECASE,
)


def conflict_reported(result: AgentResult, values: list[Any], _: JudgeContext) -> CheckOutcome:
    """The answer states every value in `values` (value-aware, like `amount`) and says the
    sources disagree (conflict/disagree/differ/restate/revise/inconsistent/discrepancy, any
    form). Averaging the values, or quietly picking one, fails. Whether the answer *explains*
    the disagreement well is left to a `judge` rubric."""
    missing = [v for v in values if not _has_amount(result.output, v)]
    word = _CONFLICT_WORDS.search(result.output)
    ok = not missing and word is not None
    found = word.group(0) if word else None
    return CheckOutcome(
        ok, f"conflict_reported {values}: missing {missing}, conflict word {found!r}"
    )


_JUDGE_SYSTEM = (
    "You are a strict grader. You will be given a task prompt, the tool outputs the model was "
    "shown (its evidence, with [n] citation numbers), the model's answer, and a rubric. A claim "
    "the evidence supports is grounded, even if it sounds like general knowledge; a claim the "
    "evidence does not support is an outside fact. "
    "Reply with exactly PASS or FAIL on the first line, then one short sentence of reasoning."
)
EVIDENCE_CHARS = 12_000
"""Cap on tool output passed to the judge, so a long filing cannot blow up a grading call."""


def evidence(result: AgentResult, limit: int = EVIDENCE_CHARS) -> str:
    """The tool outputs the model saw, in order, as one block for the judge. Truncated to
    `limit` characters with a marker, so the judge knows it is not seeing everything."""
    parts = [
        f"[{tr.get('name', 'tool')}]\n{tr.get('content', '')}"
        for step in result.steps
        for tr in step.tool_results
    ]
    text = "\n\n".join(parts) or "(no tools were called)"
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"
    return text


def judge(result: AgentResult, rubric: str, ctx: JudgeContext) -> CheckOutcome:
    if ctx.provider is None:
        return CheckOutcome(True, "judge: skipped (no judge provider)")
    user = (
        f"TASK PROMPT:\n{ctx.prompt}\n\nEVIDENCE (tool outputs the model saw):\n"
        f"{evidence(result)}\n\nMODEL ANSWER:\n{result.output}\n\n"
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
    "amount": amount,
    "no_amount": no_amount,
    "date": date,
    "cites_one_of": cites_one_of,
    "conflict_reported": conflict_reported,
    "judge": judge,
}
