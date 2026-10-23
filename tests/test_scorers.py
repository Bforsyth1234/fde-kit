"""Value-aware eval checks: they must accept every honest phrasing of a number and still catch
wrong numbers and leaks written in a different format."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.scorers import (
    JudgeContext,
    amount,
    amounts,
    conflict_reported,
    date,
    dates,
    no_amount,
)

CTX = JudgeContext(provider=None, prompt="")


def r(text: str) -> SimpleNamespace:
    return SimpleNamespace(output=text)


def test_amounts_parses_scales_and_skips_citations_and_unit_names():
    got = amounts("$51.750 billion [3], USD 16,071,700,000, Unit 3B, $4.1B, 747 million, 5%")
    assert got == pytest.approx([51.75e9, 16_071_700_000, 3, 4.1e9, 747e6, 5])


@pytest.mark.parametrize(
    "text",
    [
        "$16,071.7 million",
        "about $16.07 billion",
        "USD 16,071,700,000",
        "$16.1 billion",
    ],
)
def test_amount_matches_any_honest_phrasing(text):
    assert amount(r(text), "$16,071.7 million", CTX).passed


@pytest.mark.parametrize("text", ["$16.0 billion", "$1,607.17 million", "16,071.7"])
def test_amount_rejects_wrong_value_or_wrong_scale(text):
    # "16,071.7" with no scale is 16 thousand, not 16 billion: the scale matters.
    assert not amount(r(text), "$16,071.7 million", CTX).passed


def test_amount_tolerance_still_separates_close_restatements():
    # Pfizer FY2019 restated 41,172 -> 40,905 (0.65% apart): one must not satisfy the other.
    assert not amount(r("$40.905 billion"), "$41,172 million", CTX).passed
    assert amount(r("$41.17 billion"), "$41,172 million", CTX).passed


@pytest.mark.parametrize(
    "leak", ["$81,288 million", "$81.3 billion", "81,288,000,000", "roughly $81.29B"]
)
def test_no_amount_catches_a_leak_in_any_format(leak):
    assert not no_amount(r(f"Pfizer's revenue was {leak}."), "$81,288 million", CTX).passed


def test_no_amount_passes_a_refusal_with_other_numbers():
    text = "I couldn't find a Unit 2B. Units on file are Apartments 101–103 and 201–203 (2026)."
    assert no_amount(r(text), "$2,050", CTX).passed


def test_dates_reads_iso_and_written_forms():
    assert dates("December 1, 2026 to Nov. 30th, 2027; also 1 Dec 2026 and 2026-12-01") == {
        "2026-12-01",
        "2027-11-30",
    }
    assert date(r("from December 1, 2026"), "2026-12-01", CTX).passed
    assert not date(r("from December 2, 2026"), "2026-12-01", CTX).passed


def test_conflict_reported_is_value_aware():
    values = ["$51,750 million", "$41,172 million", "$40,905 million"]
    live = "The filings conflict: $51.750 billion [3], $41.172 billion [2], $40.905 billion [1]."
    assert conflict_reported(r(live), values, CTX).passed
    # One value dropped, or no disagreement stated: fail.
    assert not conflict_reported(r(live.replace(", $40.905 billion [1]", "")), values, CTX).passed
    quiet = "Revenue was $51.750 billion, $41.172 billion and $40.905 billion."
    assert not conflict_reported(r(quiet), values, CTX).passed
    # Small values (lease fees) still match as plain dollars.
    assert conflict_reported(r("The sources disagree: $75 vs $85."), ["75", "85"], CTX).passed


def test_amount_check_value_must_hold_one_number():
    with pytest.raises(ValueError):
        amount(r("x"), "$1 or $2", CTX)
