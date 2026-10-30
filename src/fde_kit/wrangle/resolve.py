"""Entity resolution (§7.3 of docs/pr5-wrangle-core.md): a cleaned cell value -> a canonical name.

For a part with a crosswalk, in order:

1. **exact** on the cleaned value against canonicals and aliases;
2. **normalised**: exact on `norm_key`. A key that maps to two canonicals is an
   `ambiguous entity` reject (and `check_crosswalk` reports the crosswalk itself as faulty);
3. **fuzzy** (only with `fuzzy: true`): rapidfuzz `token_set_ratio` on `norm_key`, accepted
   only when the best canonical scores >= `threshold` **and** beats the runner-up canonical by
   at least `FUZZY_MARGIN`. Accepted matches are kept for `crosswalk.proposed.csv`;
4. otherwise `closed` -> `unknown entity`; `open` -> the cleaned value is its own canonical
   (strategy `new`). With a shared `names` map (apply passes one per load, seeded with the
   crosswalk canonicals), an open value whose casefold is already in it takes that spelling
   instead (strategy `normalised`), because the store resolves names case-insensitively and
   would otherwise attach both spellings' facts to one entity anyway.

A part with no crosswalk is open: every non-empty value is `new` (or `normalised`, as above).
An empty value never resolves.
False merges cost more than misses, which is why fuzzy is off by default and the margin exists.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .spec import Crosswalk

__all__ = [
    "AMBIGUOUS",
    "FUZZY_MARGIN",
    "UNKNOWN",
    "FuzzyMatch",
    "Resolution",
    "Resolver",
    "Strategy",
    "check_crosswalk",
    "fuzzy_accepts",
    "norm_key",
]

Strategy = Literal["exact", "normalised", "fuzzy", "new"]
STRATEGIES: tuple[Strategy, ...] = ("exact", "normalised", "fuzzy", "new")
UNKNOWN = "unknown entity"
AMBIGUOUS = "ambiguous entity"
FUZZY_MARGIN = 5

_DROP = re.compile(r"[-_.#/,\s]+")
_DIGITS = re.compile(r"[0-9]+")


def norm_key(s: str) -> str:
    """casefold, NFKC, strip leading zeros from each digit run, then drop `- _ . # / ,` and
    whitespace. `DET-07` -> `det7`, `Plant 07` -> `plant7`, `1007` -> `1007`, `000` -> `0`.

    Zeros go first, while separators still delimit the runs: `10-07` -> `107` and `100-7` ->
    `1007` stay distinct. Dropping separators first would merge both into `1007`, a false merge
    of two different codes, which is the worst failure this step can have."""
    s = unicodedata.normalize("NFKC", s.casefold())
    s = _DIGITS.sub(lambda m: m.group().lstrip("0") or "0", s)
    return _DROP.sub("", s)


def fuzzy_accepts(best: float, second: float, threshold: float) -> bool:
    """The §7.3 rule: best >= threshold and best - second >= FUZZY_MARGIN."""
    return best >= threshold and best - second >= FUZZY_MARGIN


def _token_set_ratio(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return float(fuzz.token_set_ratio(a, b))


@dataclass(frozen=True)
class Resolution:
    value: str
    """The cleaned input value."""
    canonical: str | None
    """None when the value didn't resolve (see `reason`)."""
    strategy: Strategy | None
    reason: str | None = None
    """`unknown entity` or `ambiguous entity` when `canonical` is None."""
    score: float | None = None
    """Fuzzy only: the best canonical's score and the runner-up's."""
    second: float | None = None

    @property
    def ok(self) -> bool:
        return self.canonical is not None


@dataclass(frozen=True)
class FuzzyMatch:
    """One accepted fuzzy match, for review in `crosswalk.proposed.csv`."""

    crosswalk: str
    alias: str
    canonical: str
    score: float
    second: float


@dataclass
class Resolver:
    """Resolves one entity part. Results are cached per cleaned value, so `counts` and
    `fuzzy_matches` are per distinct value, not per row."""

    crosswalk: Crosswalk | None = None
    policy: Literal["closed", "open"] = "open"
    fuzzy: bool = False
    threshold: float = 92
    scorer: Callable[[str, str], float] = _token_set_ratio
    """Similarity in 0..100 of two norm_keys; injectable for tests."""
    names: dict[str, str] | None = None
    """casefold -> first spelling, shared across a load's parts; see the module docstring."""
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STRATEGIES, 0))
    fuzzy_matches: list[FuzzyMatch] = field(default_factory=list)
    _cache: dict[str, Resolution] = field(default_factory=dict, repr=False)
    _norm: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.crosswalk is None:
            self.policy = "open"
            return
        for key, canonical in self.crosswalk.mapping.items():
            nk = norm_key(key)
            if nk:
                self._norm.setdefault(nk, set()).add(canonical)

    @classmethod
    def for_part(
        cls, resolve, crosswalks: dict[str, Crosswalk], *, names: dict[str, str] | None = None
    ) -> Resolver:
        """From a spec `Resolve` block and the loaded crosswalks."""
        cw = crosswalks[resolve.crosswalk] if resolve.crosswalk else None
        return cls(
            crosswalk=cw,
            policy=resolve.policy or ("closed" if cw else "open"),
            fuzzy=resolve.fuzzy,
            threshold=resolve.threshold,
            names=names,
        )

    def resolve(self, value: str) -> Resolution:
        hit = self._cache.get(value)
        if hit is None:
            hit = self._resolve(value)
            self._cache[value] = hit
            if hit.strategy is not None:
                self.counts[hit.strategy] += 1
                if hit.strategy == "fuzzy":
                    assert self.crosswalk is not None and hit.canonical is not None
                    self.fuzzy_matches.append(
                        FuzzyMatch(
                            self.crosswalk.name,
                            value,
                            hit.canonical,
                            hit.score or 0.0,
                            hit.second or 0.0,
                        )
                    )
        return hit

    def _resolve(self, value: str) -> Resolution:
        if not value.strip():
            return Resolution(value, None, None, UNKNOWN)
        cw = self.crosswalk
        if cw is None:
            return self._open(value)
        exact = cw.mapping.get(value)
        if exact is not None:
            return Resolution(value, exact, "exact")
        nk = norm_key(value)
        cands = self._norm.get(nk, set())
        if len(cands) == 1:
            return Resolution(value, next(iter(cands)), "normalised")
        if len(cands) > 1:
            return Resolution(value, None, None, AMBIGUOUS)
        if self.fuzzy and nk:
            best_by_canonical: dict[str, float] = {}
            for key_nk, canonicals in self._norm.items():
                score = self.scorer(nk, key_nk)
                for c in canonicals:
                    if score > best_by_canonical.get(c, -1.0):
                        best_by_canonical[c] = score
            ranked = sorted(best_by_canonical.items(), key=lambda kv: (-kv[1], kv[0]))
            if ranked:
                best_c, best = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                if fuzzy_accepts(best, second, self.threshold):
                    return Resolution(value, best_c, "fuzzy", score=best, second=second)
        if self.policy == "open":
            return self._open(value)
        return Resolution(value, None, None, UNKNOWN)

    def _open(self, value: str) -> Resolution:
        if self.names is not None:
            first = self.names.setdefault(value.casefold(), value)
            if first != value:
                return Resolution(value, first, "normalised")
        return Resolution(value, value, "new")


def check_crosswalk(crosswalk: Crosswalk) -> list[str]:
    """Problems with a crosswalk on its own: keys whose `norm_key` collides across canonicals
    (each such key would be an `ambiguous entity` reject at apply). Empty when it's sound."""
    by_key: dict[str, dict[str, set[str]]] = {}
    for key, canonical in crosswalk.mapping.items():
        by_key.setdefault(norm_key(key), {}).setdefault(canonical, set()).add(key)
    problems = []
    for nk in sorted(by_key):
        groups = by_key[nk]
        if len(groups) > 1:
            keys = sorted(k for ks in groups.values() for k in ks)
            problems.append(
                f"crosswalk {crosswalk.name!r}: {', '.join(repr(k) for k in keys)} all normalise "
                f"to {nk!r} but map to different canonicals {sorted(groups)}"
            )
    return problems
