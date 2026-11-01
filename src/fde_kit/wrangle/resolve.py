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
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from .spec import Crosswalk

__all__ = [
    "AMBIGUOUS",
    "FUZZY_MARGIN",
    "SAME_BUT_PUNCTUATION",
    "SIMILAR_MAX_SCORED_NAMES",
    "SIMILAR_SPELLING",
    "UNKNOWN",
    "FuzzyMatch",
    "Resolution",
    "Resolver",
    "SimilarPair",
    "Strategy",
    "check_crosswalk",
    "fuzzy_accepts",
    "max_length_ratio",
    "norm_key",
    "similar_key",
    "similar_names",
]

Strategy = Literal["exact", "normalised", "fuzzy", "new"]
STRATEGIES: tuple[Strategy, ...] = ("exact", "normalised", "fuzzy", "new")
UNKNOWN = "unknown entity"
AMBIGUOUS = "ambiguous entity"
FUZZY_MARGIN = 5

_TOKENS = re.compile(r"([0-9]+)|([-_.#/,\s]+)|[^-_.#/,\s0-9]+")
_FRACTION = frozenset(".,")


def norm_key(s: str) -> str:
    """casefold, NFKC, strip leading zeros from each digit run, drop the separators
    `- _ . # / ,` and whitespace, except between two digits.

    `DET-07` -> `det7`, `Plant 07` -> `plant7`, `1007` -> `1007`, `000` -> `0`.

    A run of separators with a digit on both sides is kept, without its whitespace (one space
    if it is only whitespace): `10-07` -> `10-7`, `1.5` -> `1.5`, `A-1 / 10` -> `a1/10`,
    `12 345` -> `12 345`. Dropping it would merge different numbers and codes, `Pipe 1.5 in` with
    `Pipe 15 in`, `2.25` with `22.5`, `A-1/10` with `A-11/0`, `12-345` with `123-45`: a false
    merge, which is the worst failure this step can have. A digit run after a kept `.` or `,`
    keeps its leading zeros (`1.05` is not `1.5`); every other digit run loses them."""
    s = unicodedata.normalize("NFKC", s.casefold())
    tokens = [
        (m.group(), m.group(1) is not None, m.group(2) is not None) for m in _TOKENS.finditer(s)
    ]
    out: list[str] = []
    kept_sep = ""
    for i, (text, digits, sep) in enumerate(tokens):
        if digits:
            out.append(text if _FRACTION & set(kept_sep) else (text.lstrip("0") or "0"))
            kept_sep = ""
        elif sep:
            between = 0 < i < len(tokens) - 1 and tokens[i - 1][1] and tokens[i + 1][1]
            kept_sep = ("".join(text.split()) or " ") if between else ""
            out.append(kept_sep)
        else:
            out.append(text)
            kept_sep = ""
    return "".join(out)


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
SIMILAR_MAX_SCORED_NAMES = 20_000
"""Apply scores a part's names (within one scope) only up to this many distinct names; above
it the scored half is skipped and reported as `similar_skipped` (the norm_key half, a hash
grouping, always runs)."""
_SPACES = re.compile(r"\s+")
_ANY_DIGITS = re.compile(r"\d+")
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


def max_length_ratio(cutoff: float) -> float:
    """How much longer one key may be than the other and still reach `fuzz.ratio >= cutoff`.

    `ratio = 100 * 2M / (la + lb)` with M matched characters, and M <= la when la <= lb, so
    `ratio <= 200 la / (la + lb)`. That is >= cutoff only when `lb <= la (200 - cutoff) /
    cutoff`: at 90, the longer key is at most 22.2% longer. Pairs outside are never scored."""
    return (200.0 - cutoff) / cutoff


def _scored_pairs(
    keys: list[str],
    cutoff: float,
    use_cdist: bool | None,
    groups: list[str] | None = None,
) -> list[tuple[int, int, float]]:
    """(i, j, score) for i < j with `fuzz.ratio(keys[i], keys[j]) >= cutoff`, leaving out
    pairs with `groups[i] == groups[j]` when `groups` is given.

    Keys are sorted by length and each is scored only against the longer keys inside the
    `max_length_ratio` window, which drops pairs that cannot reach the cutoff before any
    scoring. Within one length, keys of one group sort together, so the extract_iter path
    skips a key's own group by slicing (a numbered series of one width is never scored
    against itself). Then `process.cdist` (C++, all cores, chunked by rows so memory is 512 x the
    window, not n x n) when numpy is importable; otherwise `process.extract_iter` per row with
    the same scorer and cutoff. Both give the same pairs; cdist needs numpy, which the wrangle
    extra doesn't pull in. `use_cdist` forces one path (tests)."""
    from rapidfuzz import fuzz, process

    if use_cdist is None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            use_cdist = False
        else:
            use_cdist = True
    n = len(keys)
    grp = groups if groups is not None else [""] * n
    order = sorted(range(n), key=lambda i: (len(keys[i]), grp[i], i))
    sk = [keys[i] for i in order]
    sg = [grp[i] for i in order]
    lens = [len(k) for k in sk]
    factor = max_length_ratio(cutoff)
    # hi[p]: one past the last position whose key may still reach the cutoff with sk[p].
    hi = [bisect_right(lens, ln * factor + 1e-9) for ln in lens]
    # own[p]: one past the run of positions with sk[p]'s length and group (p + 1 without
    # groups); every pair of p is at or after it.
    own = list(range(1, n + 1))
    if groups is not None:
        for p in range(n - 2, -1, -1):
            if lens[p] == lens[p + 1] and sg[p] == sg[p + 1]:
                own[p] = own[p + 1]
    out: list[tuple[int, int, float]] = []

    def add(p: int, q: int, score: float) -> None:
        i, j = order[p], order[q]
        out.append((i, j, score) if i < j else (j, i, score))

    if use_cdist:
        import numpy as np

        for lo in range(0, n, _CDIST_CHUNK):
            rows_end = min(lo + _CDIST_CHUNK, n)
            cols_end = hi[rows_end - 1]  # lengths ascend, so the last row's window is widest
            cols_lo = min(own[lo:rows_end])  # no row needs a column before its own group's end
            block = process.cdist(
                sk[lo:rows_end],
                sk[cols_lo:cols_end],
                scorer=fuzz.ratio,
                score_cutoff=cutoff,
                dtype=np.float32,
                workers=-1,
            )
            rows, cols = np.nonzero(block >= cutoff)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                p, q = lo + r, cols_lo + c
                if q > p and (groups is None or sg[p] != sg[q]):
                    add(p, q, float(fuzz.ratio(sk[p], sk[q])))
    else:
        for p in range(n - 1):
            start = own[p]
            for _, score, j in process.extract_iter(
                sk[p], sk[start : hi[p]], scorer=fuzz.ratio, score_cutoff=cutoff
            ):
                q = start + j
                if groups is None or sg[p] != sg[q]:
                    add(p, q, float(score))
    return sorted(out)


def _without_digits(key: str) -> str:
    return _ANY_DIGITS.sub("", key)


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
      `similar spelling`, **unless the two keys are equal once digits are removed**: names
      that differ only in their numbers (`Building 00001` / `Building 00002`, `Unit 4` /
      `Unit 5`, `Q1 sales` / `Q2 sales`) are almost always different things, and a numbered
      series would otherwise produce a pair for nearly every two members.

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
        no_digits = [_without_digits(k) for k in keys]
        for i, j, score in _scored_pairs(keys, float(threshold), use_cdist, no_digits):
            a, b = distinct[i], distinct[j]
            if (a, b) in found:
                continue
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
