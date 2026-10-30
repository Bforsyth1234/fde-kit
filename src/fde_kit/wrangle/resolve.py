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

`similar_names` is the other side of that trade: it never merges, it lists pairs of new open
names that may be one thing spelled two ways (equal `norm_key`, or `fuzz.ratio` >= a
threshold), for a person to review into a crosswalk or into `resolve.not_same`. Apply runs it
per open part after a load is resolved and reports `similar_entities`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from .spec import Crosswalk

__all__ = [
    "AMBIGUOUS",
    "FUZZY_MARGIN",
    "SAME_BUT_PUNCTUATION",
    "SIMILAR_SPELLING",
    "UNKNOWN",
    "FuzzyMatch",
    "Resolution",
    "Resolver",
    "SimilarPair",
    "Strategy",
    "check_crosswalk",
    "fuzzy_accepts",
    "norm_key",
    "similar_key",
    "similar_names",
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


SAME_BUT_PUNCTUATION = "same apart from punctuation/spacing"
SIMILAR_SPELLING = "similar spelling"
_SPACES = re.compile(r"\s+")
_CDIST_CHUNK = 512


@dataclass(frozen=True)
class SimilarPair:
    """Two distinct names of one entity part that may be the same thing. A warning only."""

    a: str
    b: str
    """`a` < `b` (plain string order)."""
    score: float
    reason: str


def similar_key(s: str) -> str:
    """What the similarity score compares: casefolded, NFKC, whitespace collapsed and stripped.
    Case is folded because the store already treats names case-insensitively; punctuation is
    kept, because `norm_key` pairs are reported separately and `Lee County`/`Lea County` must
    still score as different by one letter, not be erased into one key."""
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", s.casefold())).strip()


def _scored_pairs(
    keys: list[str], cutoff: float, use_cdist: bool | None
) -> list[tuple[int, int, float]]:
    """(i, j, score) for i < j with `fuzz.ratio(keys[i], keys[j]) >= cutoff`.

    `process.cdist` (C++, all cores, chunked by rows so memory is 512 x n, not n x n) when
    numpy is importable; otherwise `process.extract_iter` per row with the same scorer and
    cutoff. Both give the same pairs; cdist needs numpy, which the wrangle extra doesn't pull
    in. `use_cdist` forces one path (tests)."""
    from rapidfuzz import fuzz, process

    if use_cdist is None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            use_cdist = False
        else:
            use_cdist = True
    out: list[tuple[int, int, float]] = []
    n = len(keys)
    if use_cdist:
        import numpy as np

        for lo in range(0, n, _CDIST_CHUNK):
            block = process.cdist(
                keys[lo : lo + _CDIST_CHUNK],
                keys,
                scorer=fuzz.ratio,
                score_cutoff=cutoff,
                dtype=np.float32,
                workers=-1,
            )
            rows, cols = np.nonzero(block >= cutoff)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                i = lo + r
                if c > i:
                    out.append((i, c, float(fuzz.ratio(keys[i], keys[c]))))
    else:
        for i in range(n - 1):
            for _, score, j in process.extract_iter(
                keys[i], keys[i + 1 :], scorer=fuzz.ratio, score_cutoff=cutoff
            ):
                out.append((i, i + 1 + j, float(score)))
    return out


def similar_names(
    names: list[str],
    threshold: float,
    *,
    not_same: Iterable[tuple[str, str]] = (),
    use_cdist: bool | None = None,
) -> list[SimilarPair]:
    """Pairs of distinct names that may be one entity spelled two ways. Never merges anything.

    - Pairs with the same `norm_key` but different text are always reported, score 100,
      reason `same apart from punctuation/spacing` (`Beer, wine, and liquor stores` /
      `Beer, wine and liquor stores`).
    - With `threshold` > 0, pairs whose `fuzz.ratio` on `similar_key` is >= `threshold`, reason
      `similar spelling`.

    **Why `fuzz.ratio`.** It is the normalised Indel similarity of the whole strings, so every
    added, dropped or changed character costs in proportion to the length: a typo or a dropped
    comma in a long name scores in the 90s, while a name that adds a word (`Clothing stores` /
    `Men's clothing stores`) falls well below. `token_set_ratio` and `partial_ratio`, which
    suit resolving a cell against a crosswalk, score a name that is a subset of another 100,
    and real hierarchical tables (MRTS kinds of business: a category and its sub-categories)
    would then flood the report with parent/child pairs that are certainly different. `WRatio`
    mixes those in for the same reason. The cost is that short names differing by a letter
    (`Lee County` / `Lea County`, 90) do fire: those are the pairs `not_same` exists for.

    `not_same` pairs (exact strings, either order) are dropped. Sorted by score descending,
    then `a`, then `b`."""
    distinct = sorted(set(names))
    suppressed = {frozenset(p) for p in not_same}
    found: dict[tuple[str, str], SimilarPair] = {}

    by_norm: dict[str, list[str]] = {}
    for n in distinct:
        by_norm.setdefault(norm_key(n), []).append(n)
    for group in by_norm.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                found[(a, b)] = SimilarPair(a, b, 100.0, SAME_BUT_PUNCTUATION)

    if threshold > 0 and len(distinct) > 1:
        keys = [similar_key(n) for n in distinct]
        for i, j, score in _scored_pairs(keys, float(threshold), use_cdist):
            a, b = distinct[i], distinct[j]
            if (a, b) not in found:
                found[(a, b)] = SimilarPair(a, b, round(score, 1), SIMILAR_SPELLING)

    pairs = [p for p in found.values() if frozenset((p.a, p.b)) not in suppressed]
    return sorted(pairs, key=lambda p: (-p.score, p.a, p.b))


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
