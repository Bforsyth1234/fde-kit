"""`values_agree`, the precision-aware conflict rule (PR 4), on the real EDGAR numbers.

Pure: no database. The pairs are the ones that probing XBRL companyfacts turned up (see
docs/pr4-edgar-adapters.md §1): rounded repeats that must agree, restatements that must not.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("psycopg")

from fde_kit.rag import values_agree  # noqa: E402

M = 1_000_000

AGREE = [
    # Vertex FY2019 revenue: exact in two 10-Ks, to $100K in the FY2021 one.
    pytest.param(4_162_821_000, 1000, 4_162_800_000, 100_000, id="vertex-rounded-repeat"),
    # Moderna FY2020 net loss: in thousands, then in millions.
    pytest.param(-747_064_000, 1000, -747_000_000, M, id="moderna-net-loss-rounded"),
    # Identical values agree at any precision, including none.
    pytest.param(51_750 * M, M, 51_750 * M, M, id="pfizer-same-filing-value"),
    pytest.param(1800, None, 1800, None, id="exact-equal"),
    # Half-up edge: 1850 / 100 = 18.5 -> 19, 1900 / 100 = 19.
    pytest.param(1850, 100, 1900, 100, id="half-up-edge"),
    # Half away from zero: -18.5 -> -19, mirror of the positive case.
    pytest.param(-1850, 100, -1900, 100, id="half-away-from-zero-negative"),
    # One side exact, the other coarse: compared at the coarse one.
    pytest.param(4_162_821_000, None, 4_162_800_000, 100_000, id="exact-vs-coarse"),
    # A float and a Decimal of the same value (what comes back from NUMERIC).
    pytest.param(0.1, None, Decimal("0.1"), None, id="float-vs-decimal"),
]

DISAGREE = [
    # Pfizer FY2019 revenue: FY2019 10-K, then the Upjohn recast, then the FY2021 10-K.
    pytest.param(51_750 * M, M, 41_172 * M, M, id="pfizer-upjohn-recast"),
    pytest.param(41_172 * M, M, 40_905 * M, M, id="pfizer-second-restatement"),
    pytest.param(51_750 * M, M, 40_905 * M, M, id="pfizer-first-vs-third"),
    # Gilead FY2019 R&D.
    pytest.param(9_106 * M, M, 4_055 * M, M, id="gilead-rnd"),
    # Regeneron FY2019 revenue and R&D.
    pytest.param(7_863 * M, M, 6_558 * M, M, id="regeneron-revenue"),
    pytest.param(3_037 * M, M, 2_450 * M, M, id="regeneron-rnd"),
    # No blanket tolerance: rents with no precision are exact.
    pytest.param(1800, None, 1850, None, id="rent-1800-vs-1850"),
    pytest.param(50, None, 60, None, id="50-vs-60"),
    pytest.param(1800, 1, 1850, 1, id="rent-at-precision-1"),
    # Both precisions None is exact, not "rounded to units".
    pytest.param(1.2, None, 1.4, None, id="exact-decimals"),
    # Sign matters: -1850 and 1850 round to -19 and 19.
    pytest.param(-1850, 100, 1850, 100, id="sign"),
    # Half-up edge the other way: 1849 / 100 -> 18, 1850 / 100 -> 19.
    pytest.param(1849, 100, 1850, 100, id="just-below-half"),
]


@pytest.mark.parametrize(("a", "pa", "b", "pb"), AGREE)
def test_agree(a, pa, b, pb):
    assert values_agree(a, pa, b, pb)
    assert values_agree(b, pb, a, pa)  # symmetric


@pytest.mark.parametrize(("a", "pa", "b", "pb"), DISAGREE)
def test_disagree(a, pa, b, pb):
    assert not values_agree(a, pa, b, pb)
    assert not values_agree(b, pb, a, pa)


def test_text_agrees_iff_equal():
    assert values_agree("waived", None, "waived", None)
    assert not values_agree("waived", None, "not applicable", None)
    assert not values_agree("Waived", None, "waived", None)


def test_number_and_text_never_agree():
    assert not values_agree(1800, None, "1800", None)
    assert not values_agree("1800", None, 1800, 1000)


def test_large_integers_are_exact():
    """Floating point would lose the last digit here; Decimal does not."""
    big = 10**20 + 1
    assert not values_agree(big, None, big + 1, None)
    assert values_agree(big, 10, big + 1, 10)
    # (10**20 + 4) / 10 ends in .4 and rounds down; (10**20 + 5) / 10 ends in .5 and rounds up.
    assert not values_agree(big + 3, 10, big + 4, 10)
    assert values_agree(big + 4, 10, big + 5, 10)  # .5 and .6 both round up


def test_non_positive_precision_is_rejected():
    with pytest.raises(ValueError):
        values_agree(1, 0, 1, None)
