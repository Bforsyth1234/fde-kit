"""Entity resolution (§7.3): norm_key, each strategy, ambiguity, the fuzzy margin rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from fde_kit.wrangle.resolve import (
    AMBIGUOUS,
    UNKNOWN,
    Resolver,
    check_crosswalk,
    fuzzy_accepts,
    norm_key,
)
from fde_kit.wrangle.spec import Crosswalk, Resolve, load_project_spec

FIXTURES = Path(__file__).parent / "fixtures" / "wrangle"


def _crosswalk(pairs: list[tuple[str, str]], name: str = "plants") -> Crosswalk:
    canonicals = frozenset(c for _, c in pairs)
    mapping = {c: c for c in canonicals}
    mapping.update(dict(pairs))
    return Crosswalk(name, "plant", Path(f"{name}.csv"), mapping, canonicals, "0" * 64)


PLANTS = _crosswalk(
    [
        ("1007", "Detroit Stamping 07"),
        ("DET-07", "Detroit Stamping 07"),
        ("Detroit Plt 07", "Detroit Stamping 07"),
        ("1012", "Toledo Assembly 12"),
        ("TOL-12", "Toledo Assembly 12"),
        ("Toledo Plt 12", "Toledo Assembly 12"),
    ]
)


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("DET-07", "det7"),
        ("Plant 07", "plant7"),
        ("1007", "1007"),
        ("000", "0"),
        # Separators go first, so digit runs either side of one merge: one run, one strip.
        ("0012-007", "127"),
        ("Plt 012 / 0", "plt120"),
        ("A_b.c#d/e,f g", "abcdefg"),
        ("TOLEDO PLT 12", "toledoplt12"),
        ("  Toledo\tPlt   12 ", "toledoplt12"),
        ("ＤＥＴ－０７", "det7"),  # fullwidth, folded by NFKC
        ("Straße", "strasse"),  # casefold, not lower
        ("", ""),
    ],
)
def test_norm_key(raw: str, key: str) -> None:
    assert norm_key(raw) == key


def test_exact_on_alias_and_on_canonical() -> None:
    r = Resolver(PLANTS, "closed")
    a = r.resolve("DET-07")
    assert (a.canonical, a.strategy) == ("Detroit Stamping 07", "exact")
    b = r.resolve("Toledo Assembly 12")
    assert (b.canonical, b.strategy) == ("Toledo Assembly 12", "exact")


def test_normalised_match() -> None:
    r = Resolver(PLANTS, "closed")
    for raw in ("TOLEDO PLT 12", "det 7", "Det_007", "detroit plt 7"):
        res = r.resolve(raw)
        assert res.strategy == "normalised", raw
    assert r.resolve("TOLEDO PLT 12").canonical == "Toledo Assembly 12"
    assert r.resolve("det 7").canonical == "Detroit Stamping 07"


def test_closed_rejects_unknown_and_open_keeps_it() -> None:
    closed = Resolver(PLANTS, "closed").resolve("GRR-03")
    assert closed.canonical is None and closed.reason == UNKNOWN and not closed.ok
    opened = Resolver(PLANTS, "open").resolve("GRR-03")
    assert (opened.canonical, opened.strategy) == ("GRR-03", "new")


def test_no_crosswalk_is_open() -> None:
    r = Resolver(None, "closed")
    assert r.policy == "open"
    res = r.resolve("P-1234")
    assert (res.canonical, res.strategy) == ("P-1234", "new")


def test_empty_value_never_resolves() -> None:
    for r in (Resolver(None), Resolver(PLANTS, "open"), Resolver(PLANTS, "closed")):
        res = r.resolve("  ")
        assert res.canonical is None and res.reason == UNKNOWN


def test_ambiguous_norm_key_collision() -> None:
    cw = _crosswalk([("DET-07", "Detroit Stamping 07"), ("det 7", "Detroit Trim 07")])
    r = Resolver(cw, "open", fuzzy=True)  # neither open nor fuzzy rescues an ambiguous key
    res = r.resolve("Det07")
    assert res.canonical is None and res.reason == AMBIGUOUS
    # The exact spelling still resolves: step 1 runs before the norm_key step.
    assert r.resolve("DET-07").canonical == "Detroit Stamping 07"
    problems = check_crosswalk(cw)
    assert len(problems) == 1
    assert "det7" in problems[0] and "Detroit Trim 07" in problems[0]


def test_check_crosswalk_clean_on_fixture() -> None:
    loaded = load_project_spec(FIXTURES / "project" / "specs" / "planner_forecast.yaml")
    assert check_crosswalk(loaded.crosswalks["plants"]) == []


def test_aliases_of_one_canonical_do_not_collide() -> None:
    cw = _crosswalk([("DET-07", "Detroit Stamping 07"), ("det 7", "Detroit Stamping 07")])
    assert check_crosswalk(cw) == []


@pytest.mark.parametrize(
    ("best", "second", "threshold", "ok"),
    [
        (92, 90, 92, False),  # over threshold, margin 2
        (95, 85, 92, True),
        (97, 92, 92, True),  # margin exactly 5
        (97, 93, 92, False),
        (91, 0, 92, False),  # under threshold
        (92, 87, 92, True),
        (80, 0, 75, True),
    ],
)
def test_fuzzy_margin_rule(best: float, second: float, threshold: float, ok: bool) -> None:
    assert fuzzy_accepts(best, second, threshold) is ok


def _scorer(scores: dict[str, float]):
    """Score a key by the canonical its norm_key belongs to (both Alpha keys score alike)."""

    def score(_value: str, key: str) -> float:
        return scores.get(key, 0.0)

    return score


ALPHA_BETA = _crosswalk([("A-1", "Alpha"), ("B-1", "Beta")])


def test_fuzzy_resolver_92_90_rejects() -> None:
    r = Resolver(
        ALPHA_BETA, "closed", fuzzy=True, scorer=_scorer({"a1": 92, "alpha": 92, "b1": 90})
    )
    res = r.resolve("zzz")
    assert res.canonical is None and res.reason == UNKNOWN
    assert r.fuzzy_matches == []


def test_fuzzy_resolver_95_85_accepts_and_proposes() -> None:
    r = Resolver(ALPHA_BETA, "closed", fuzzy=True, scorer=_scorer({"a1": 95, "b1": 85, "beta": 80}))
    res = r.resolve("zzz")
    assert (res.canonical, res.strategy, res.score, res.second) == ("Alpha", "fuzzy", 95, 85)
    r.resolve("zzz")  # cached: counted and proposed once
    assert r.counts["fuzzy"] == 1
    assert [(m.crosswalk, m.alias, m.canonical, m.score, m.second) for m in r.fuzzy_matches] == [
        ("plants", "zzz", "Alpha", 95, 85)
    ]


def test_fuzzy_margin_is_between_canonicals_not_keys() -> None:
    # Two keys of the same canonical score 96 and 95: the runner-up is Beta's 60, so accept.
    r = Resolver(
        ALPHA_BETA, "closed", fuzzy=True, scorer=_scorer({"a1": 96, "alpha": 95, "b1": 60})
    )
    assert r.resolve("zzz").canonical == "Alpha"


def test_fuzzy_threshold_is_configurable() -> None:
    r = Resolver(ALPHA_BETA, "closed", fuzzy=True, threshold=80, scorer=_scorer({"a1": 85}))
    assert r.resolve("zzz").strategy == "fuzzy"


def test_fuzzy_with_rapidfuzz() -> None:
    r = Resolver(PLANTS, "closed", fuzzy=True)
    res = r.resolve("Detroit Stampng 07")  # a dropped letter
    assert (res.canonical, res.strategy) == ("Detroit Stamping 07", "fuzzy")
    assert res.score is not None and res.score >= 92
    off = Resolver(PLANTS, "closed", fuzzy=False).resolve("Detroit Stampng 07")
    assert off.reason == UNKNOWN


def test_fuzzy_miss_falls_back_to_policy() -> None:
    closed = Resolver(PLANTS, "closed", fuzzy=True).resolve("Grand Rapids 03")
    assert closed.reason == UNKNOWN
    opened = Resolver(PLANTS, "open", fuzzy=True).resolve("Grand Rapids 03")
    assert (opened.canonical, opened.strategy) == ("Grand Rapids 03", "new")


def test_counts_are_per_distinct_value() -> None:
    r = Resolver(PLANTS, "closed")
    for raw in ("DET-07", "DET-07", "TOLEDO PLT 12", "GRR-03", "1012"):
        r.resolve(raw)
    assert r.counts == {"exact": 2, "normalised": 1, "fuzzy": 0, "new": 0}


def test_for_part_reads_spec_resolve_block() -> None:
    loaded = load_project_spec(FIXTURES / "project" / "specs" / "planner_forecast.yaml")
    site, part = loaded.spec.entity.parts
    rs = Resolver.for_part(site.resolve, loaded.crosswalks)
    assert rs.crosswalk is loaded.crosswalks["plants"] and rs.policy == "closed"
    assert rs.fuzzy is False and rs.threshold == 92
    rp = Resolver.for_part(part.resolve, loaded.crosswalks)
    assert rp.crosswalk is None and rp.policy == "open"
    # A crosswalk with no policy defaults to closed (spec.Resolve's own default).
    d = Resolver.for_part(Resolve(crosswalk="plants"), loaded.crosswalks)
    assert d.policy == "closed"


def test_norm_key_strips_zeros_before_dropping_separators() -> None:
    # Merging digit runs first would turn both into "1007": a false merge of two codes.
    assert norm_key("10-07") == "107"
    assert norm_key("100-7") == "1007"
    assert norm_key("10-07") != norm_key("100-7")
