"""Citation numbering and parsing, shared by the in-process loop and the durable workflow.

Contract:

- A tool that cites returns `ToolResult(content, sources=(s1, s2, ...))`. Inside `content` it
  refers to its own sources with local markers `[#1]`, `[#2]`, ... (1-based into `sources`).
- The loop keeps one `CitationLedger` per run. Each new Source gets the next run-wide number;
  a Source seen twice (same `source_id`) keeps its first number. Local markers in the tool's
  content are rewritten to the run-wide `[n]` before the content reaches the model.
- The model is asked to cite by `[n]`. `cited(output, ledger)` resolves the numbers that appear
  in the final answer back to Sources. Numbers that do not exist in the ledger are ignored: a
  fabricated `[9]` with only three sources is not a citation, and the `cites_only` eval check
  counts it against the run.

Pure functions and one small class: no I/O, no clocks, so the Temporal workflow can import it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tools import Source

_LOCAL = re.compile(r"\[#(\d+)\]")
# [3]  [3, 5]  [3][5]  — one or more comma/space-separated integers inside square brackets.
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class CitationLedger:
    sources: list[Source] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)  # source_id -> 1-based number

    def number_of(self, s: Source) -> int:
        n = self._index.get(s.source_id)
        if n is None:
            self.sources.append(s)
            n = len(self.sources)
            self._index[s.source_id] = n
        return n

    def add(
        self, content: str, sources: tuple[Source, ...] | list[Source]
    ) -> tuple[str, list[int]]:
        """Register `sources`, rewrite `[#i]` markers in `content` to run-wide numbers.

        Returns (rewritten content, run-wide numbers in the order of `sources`).
        """
        numbers = [self.number_of(s) for s in sources]
        if not numbers:
            return content, numbers

        def repl(m: re.Match[str]) -> str:
            i = int(m.group(1))
            if 1 <= i <= len(numbers):
                return f"[{numbers[i - 1]}]"
            return m.group(0)

        return _LOCAL.sub(repl, content), numbers

    def resolve(self, n: int) -> Source | None:
        return self.sources[n - 1] if 1 <= n <= len(self.sources) else None


def cited_numbers(output: str) -> list[int]:
    """Every `[n]` citation number in `output`, in first-seen order, deduplicated."""
    seen: list[int] = []
    for m in _CITE.finditer(output):
        for part in m.group(1).split(","):
            n = int(part.strip())
            if n not in seen:
                seen.append(n)
    return seen


def cited(output: str, sources: list[Source]) -> list[Source]:
    """Sources the output actually cites, resolved by number. Unknown numbers are dropped."""
    out: list[Source] = []
    for n in cited_numbers(output):
        if 1 <= n <= len(sources) and sources[n - 1] not in out:
            out.append(sources[n - 1])
    return out


def unresolved_citations(output: str, sources: list[Source]) -> list[int]:
    """Citation numbers in `output` that point at no Source: fabricated citations."""
    return [n for n in cited_numbers(output) if not (1 <= n <= len(sources))]


CITE_INSTRUCTION = (
    "Some tool results are numbered passages like [1], [2]. When your answer relies on one, "
    "cite its number in square brackets, e.g. 'The late fee is $50 [2].' Cite only numbers "
    "that appear in the tool results."
)
