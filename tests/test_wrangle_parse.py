"""Parse ops (§6): strict numbers, dates, fiscal periods, and spec-vs-data error separation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from fde_kit.wrangle.parse import (
    ParseError,
    apply_map,
    apply_regex,
    apply_text,
    compile_ops,
    constant,
    parse_date,
    parse_number,
    parse_period,
    period_of,
    run_ops,
    to_fact_value,
    validate_ops,
)

D = Decimal
US = {"thousands": ",", "decimal": "."}
EU = {"thousands": ".", "decimal": ","}
CH = {"thousands": "'", "decimal": "."}
FR = {"thousands": " ", "decimal": ","}
NONE = {"thousands": None, "decimal": "."}

# ------------------------------------------------------------------------------- number

NUMBER_OK = [
    ("0", {}, D("0")),
    ("1234", {}, D("1234")),
    ("1,234", {}, D("1234")),
    ("1,234.5", {}, D("1234.5")),
    ("1,234,567.89", {}, D("1234567.89")),
    ("  42  ", {}, D("42")),
    ("-42", {}, D("-42")),
    ("+42", {}, D("42")),
    ("−42", {}, D("-42")),
    (".5", {}, D("0.5")),
    ("0.10", {}, D("0.10")),
    ("1234.50", {}, D("1234.50")),
    ("$1,234", {}, D("1234")),
    ("-$1,234", {}, D("-1234")),
    ("$-1,234", {}, D("-1234")),
    ("€ 1,234.5", {}, D("1234.5")),
    ("£12", {}, D("12")),
    ("¥1000", {}, D("1000")),
    ("1,234 USD", {}, D("1234")),
    ("USD 1,234", {}, D("1234")),
    ("EUR1234.5", {}, D("1234.5")),
    ("1.234,56", EU, D("1234.56")),
    ("1.234.567,8", EU, D("1234567.8")),
    ("1234,5", EU, D("1234.5")),
    ("1,234", EU, D("1.234")),
    ("€1.234,56", EU, D("1234.56")),
    ("1'234.5", CH, D("1234.5")),
    ("1’234.5", CH, D("1234.5")),
    ("1 234,5", FR, D("1234.5")),
    ("1 234,5", FR, D("1234.5")),
    ("1 234 567", FR, D("1234567")),
    ("1234567.25", NONE, D("1234567.25")),
    ("(1,234)", {"parens_negative": True}, D("-1234")),
    ("($1,234.50)", {"parens_negative": True}, D("-1234.50")),
    ("$(1,234.50)", {"parens_negative": True}, D("-1234.50")),
    ("( 12 )", {"parens_negative": True}, D("-12")),
    ("1,234-", {"trailing_minus": True}, D("-1234")),
    ("1.234,5-", {**EU, "trailing_minus": True}, D("-1234.5")),
    ("12%", {"percent": True}, D("0.12")),
    ("12.5%", {"percent": True}, D("0.125")),
    ("-3 %", {"percent": True}, D("-0.03")),
    ("(4%)", {"percent": True, "parens_negative": True}, D("-0.04")),
    ("0.12", {"percent": True}, D("0.12")),
    ("100%", {"percent": True}, D("1.00")),
    ("1234.0", {}, D("1234.0")),
    ("12345678901234567890.123", {}, D("12345678901234567890.123")),
]

NUMBER_REJECT = [
    ("", {}, "number: empty"),
    ("   ", {}, "number: empty"),
    ("abc", {}, "number: unparseable"),
    ("1,234.5", EU, "number: unparseable"),
    ("1.234,5", US, "number: unparseable"),
    ("1,23", {}, "number: unparseable"),
    ("1,2345", {}, "number: unparseable"),
    ("12,34,567", {}, "number: unparseable"),
    (",123", {}, "number: unparseable"),
    ("1.", {}, "number: unparseable"),
    ("1..2", {}, "number: unparseable"),
    ("1,234", NONE, "number: unparseable"),
    ("12%", {}, "number: percent not allowed"),
    ("(12)", {}, "number: parentheses not allowed"),
    ("12-", {}, "number: unparseable"),
    ("-(12)", {"parens_negative": True}, "number: unparseable"),
    ("(-12)", {"parens_negative": True}, "number: double negative"),
    ("-12-", {"trailing_minus": True}, "number: double negative"),
    ("$12", {"currency": False}, "number: unparseable"),
    ("12 USD", {"currency": False}, "number: unparseable"),
    ("$$12", {}, "number: unparseable"),
    ("1e5", {}, "number: unparseable"),
    ("1E+5", {}, "number: unparseable"),
    ("0x10", {}, "number: unparseable"),
    ("NaN", {}, "number: unparseable"),
    ("Infinity", {}, "number: unparseable"),
    ("1 234", {}, "number: unparseable"),
    ("12 34", FR, "number: unparseable"),
    ("12abc", {}, "number: unparseable"),
    ("--5", {}, "number: unparseable"),
    ("n/a", {}, "number: unparseable"),
    ("-", {}, "number: unparseable"),
    ("1'234", US, "number: unparseable"),
]


@pytest.mark.parametrize(("raw", "params", "want"), NUMBER_OK)
def test_number_ok(raw, params, want):
    got = parse_number(raw, **params)
    assert got == want
    assert isinstance(got, Decimal)
    assert run_ops(raw, [{"op": "number", **params}]) == want


@pytest.mark.parametrize(("raw", "params", "reason"), NUMBER_REJECT)
def test_number_reject(raw, params, reason):
    with pytest.raises(ParseError) as e:
        parse_number(raw, **params)
    assert e.value.reason == reason
    with pytest.raises(ParseError) as e2:
        run_ops(raw, [{"op": "number", **params}])
    assert e2.value.op == "number"


def test_number_keeps_exact_decimal_digits():
    assert str(parse_number("0.1")) == "0.1"
    assert str(parse_number("1,234.500")) == "1234.500"


# --- round trips: format a Decimal in each declared style, parse it back --------------


def fmt(d: Decimal, *, thousands, decimal, negative_style="minus", cur="", pct=False) -> str:
    v = abs(d) * (100 if pct else 1)
    s = format(v, "f")
    int_part, _, frac = s.partition(".")
    if thousands:
        groups = []
        while len(int_part) > 3:
            groups.insert(0, int_part[-3:])
            int_part = int_part[:-3]
        groups.insert(0, int_part)
        int_part = thousands.join(groups)
    body = int_part + (decimal + frac if frac else "")
    if pct:
        body += "%"
    body = cur + body
    if d < 0:
        if negative_style == "minus":
            body = "-" + body
        elif negative_style == "parens":
            body = f"({body})"
        else:
            body = body + "-"
    return body


ROUND_TRIP_VALUES = [
    D("0"),
    D("7"),
    D("999"),
    D("1000"),
    D("1234.5"),
    D("-1234.56"),
    D("1234567.891"),
    D("-1000000"),
    D("0.001"),
    D("-0.5"),
    D("98765432109876.54321"),
]
STYLES = [
    {"thousands": ",", "decimal": "."},
    {"thousands": ".", "decimal": ","},
    {"thousands": "'", "decimal": "."},
    {"thousands": " ", "decimal": ","},
    {"thousands": None, "decimal": "."},
    {"thousands": None, "decimal": ","},
]


@pytest.mark.parametrize("style", STYLES, ids=lambda s: f"{s['thousands']}|{s['decimal']}")
@pytest.mark.parametrize("negative_style", ["minus", "parens", "trailing"])
@pytest.mark.parametrize("cur", ["", "$", "EUR "])
def test_number_round_trip(style, negative_style, cur):
    params = {
        **style,
        "parens_negative": negative_style == "parens",
        "trailing_minus": negative_style == "trailing",
    }
    for d in ROUND_TRIP_VALUES:
        text = fmt(d, negative_style=negative_style, cur=cur, **style)
        assert parse_number(text, **params) == d, text


@pytest.mark.parametrize("style", STYLES, ids=lambda s: f"{s['thousands']}|{s['decimal']}")
def test_number_round_trip_percent(style):
    for d in [D("0.12"), D("-0.035"), D("1.5"), D("12.3456")]:
        text = fmt(d, pct=True, **style)
        assert parse_number(text, percent=True, **style) == d, text


def test_to_fact_value():
    assert to_fact_value(D("1234")) == 1234 and isinstance(to_fact_value(D("1234")), int)
    assert isinstance(to_fact_value(D("1234.000")), int)
    assert to_fact_value(D("1E+3")) == 1000 and isinstance(to_fact_value(D("1E+3")), int)
    assert to_fact_value(D("-0")) == 0
    assert to_fact_value(D("0.1")) == 0.1 and isinstance(to_fact_value(D("0.1")), float)
    assert repr(to_fact_value(D("1234.56"))) == "1234.56"
    assert to_fact_value(parse_number("1,234.4") * 1000) == 1234400
    with pytest.raises(ValueError):
        to_fact_value(D("NaN"))


# --------------------------------------------------------------------------------- date

DATE_OK = [
    ("2026-03-04", {"format": "%Y-%m-%d"}, date(2026, 3, 4)),
    (" 2026-03-04 ", {"format": "%Y-%m-%d"}, date(2026, 3, 4)),
    ("2026-03-04T13:45:00", {"format": "%Y-%m-%dT%H:%M:%S"}, date(2026, 3, 4)),
    ("04/03/2026", {"format": "%d/%m/%Y"}, date(2026, 3, 4)),
    ("03/04/2026", {"format": "%m/%d/%Y"}, date(2026, 3, 4)),
    ("4/3/2026", {"format": "%d/%m/%Y"}, date(2026, 3, 4)),
    ("31/12/2025", {"format": "%d/%m/%Y"}, date(2025, 12, 31)),
    ("29/02/2024", {"format": "%d/%m/%Y"}, date(2024, 2, 29)),
    ("04.03.2026", {"format": "%d.%m.%Y"}, date(2026, 3, 4)),
    ("04-Mar-2026", {"format": "%d-%b-%Y"}, date(2026, 3, 4)),
    ("4 March 2026", {"format": "%d %B %Y"}, date(2026, 3, 4)),
    ("March 4, 2026", {"format": "%B %d, %Y"}, date(2026, 3, 4)),
    ("20260304", {"format": "%Y%m%d"}, date(2026, 3, 4)),
    ("04/03/26", {"format": "%d/%m/%y"}, date(2026, 3, 4)),
    ("Mar-26", {"format": "%b-%y"}, date(2026, 3, 1)),
    ("2026/3/4", {"format": "%Y/%m/%d"}, date(2026, 3, 4)),
    ("1", {"excel_serial": True}, date(1900, 1, 1)),
    ("2", {"excel_serial": True}, date(1900, 1, 2)),
    ("31", {"excel_serial": True}, date(1900, 1, 31)),
    ("59", {"excel_serial": True}, date(1900, 2, 28)),
    ("61", {"excel_serial": True}, date(1900, 3, 1)),
    ("367", {"excel_serial": True}, date(1901, 1, 1)),
    ("36526", {"excel_serial": True}, date(2000, 1, 1)),
    ("43831", {"excel_serial": True}, date(2020, 1, 1)),
    ("45000", {"excel_serial": True}, date(2023, 3, 15)),
    ("45000.75", {"excel_serial": True}, date(2023, 3, 15)),
    ("46022", {"excel_serial": True}, date(2025, 12, 31)),
    ("46023", {"excel_serial": True}, date(2026, 1, 1)),
    (" 46023 ", {"excel_serial": True}, date(2026, 1, 1)),
    ("46023.0", {"excel_serial": True}, date(2026, 1, 1)),
    ("2958465", {"excel_serial": True}, date(9999, 12, 31)),
]

DATE_REJECT = [
    ("", {"format": "%Y-%m-%d"}, "date: empty"),
    ("2026-13-01", {"format": "%Y-%m-%d"}, "date: unparseable"),
    ("2026-02-30", {"format": "%Y-%m-%d"}, "date: unparseable"),
    ("29/02/2025", {"format": "%d/%m/%Y"}, "date: unparseable"),
    ("13/31/2026", {"format": "%m/%d/%Y"}, "date: unparseable"),
    ("2026-03-04", {"format": "%d/%m/%Y"}, "date: unparseable"),
    ("04/03/2026 extra", {"format": "%d/%m/%Y"}, "date: unparseable"),
    ("2026-03-04T13:45:00", {"format": "%Y-%m-%d"}, "date: unparseable"),
    ("yesterday", {"format": "%Y-%m-%d"}, "date: unparseable"),
    ("0", {"excel_serial": True}, "date: excel serial out of range"),
    ("-5", {"excel_serial": True}, "date: excel serial out of range"),
    ("0.5", {"excel_serial": True}, "date: excel serial out of range"),
    ("2958466", {"excel_serial": True}, "date: excel serial out of range"),
    ("60", {"excel_serial": True}, "date: excel serial is 1900-02-29, which doesn't exist"),
    ("abc", {"excel_serial": True}, "date: unparseable"),
    ("2026-01-01", {"excel_serial": True}, "date: unparseable"),
    ("NaN", {"excel_serial": True}, "date: unparseable"),
]


@pytest.mark.parametrize(("raw", "params", "want"), DATE_OK)
def test_date_ok(raw, params, want):
    assert parse_date(raw, **params) == want
    assert run_ops(raw, [{"op": "date", **params}]) == want


@pytest.mark.parametrize(("raw", "params", "reason"), DATE_REJECT)
def test_date_reject(raw, params, reason):
    with pytest.raises(ParseError) as e:
        parse_date(raw, **params)
    assert e.value.reason == reason


def test_excel_serial_agrees_with_openpyxl():
    from openpyxl.utils.datetime import from_excel

    for serial in [1, 59, 61, 100, 36526, 45000, 46023]:
        assert parse_date(str(serial), excel_serial=True) == from_excel(serial).date()


# ------------------------------------------------------------------------------- period

P = date
PERIOD_OK = [
    # format + grain: the period containing the parsed date
    ("Jan-26", {"format": "%b-%y", "grain": "month"}, (P(2026, 1, 1), P(2026, 1, 31))),
    ("Feb-24", {"format": "%b-%y", "grain": "month"}, (P(2024, 2, 1), P(2024, 2, 29))),
    ("Feb-26", {"format": "%b-%y", "grain": "month"}, (P(2026, 2, 1), P(2026, 2, 28))),
    ("Dec-26", {"format": "%b-%y", "grain": "month"}, (P(2026, 12, 1), P(2026, 12, 31))),
    ("Apr 2026", {"format": "%b %Y", "grain": "month"}, (P(2026, 4, 1), P(2026, 4, 30))),
    ("2026-03", {"format": "%Y-%m", "grain": "month"}, (P(2026, 3, 1), P(2026, 3, 31))),
    ("2026-03-17", {"format": "%Y-%m-%d", "grain": "month"}, (P(2026, 3, 1), P(2026, 3, 31))),
    ("2026-03-17", {"format": "%Y-%m-%d", "grain": "day"}, (P(2026, 3, 17), P(2026, 3, 17))),
    ("2026-03-17", {"format": "%Y-%m-%d", "grain": "instant"}, (P(2026, 3, 17), P(2026, 3, 17))),
    ("31/12/2025", {"format": "%d/%m/%Y", "grain": "instant"}, (P(2025, 12, 31), P(2025, 12, 31))),
    ("2026-03-17", {"format": "%Y-%m-%d", "grain": "quarter"}, (P(2026, 1, 1), P(2026, 3, 31))),
    ("2026-04-01", {"format": "%Y-%m-%d", "grain": "quarter"}, (P(2026, 4, 1), P(2026, 6, 30))),
    ("2026-12-31", {"format": "%Y-%m-%d", "grain": "quarter"}, (P(2026, 10, 1), P(2026, 12, 31))),
    ("2026-03-17", {"format": "%Y-%m-%d", "grain": "year"}, (P(2026, 1, 1), P(2026, 12, 31))),
    ("2026", {"format": "%Y", "grain": "year"}, (P(2026, 1, 1), P(2026, 12, 31))),
    # fiscal containment with fy_start_month 10 (FY named by the year it ends in)
    (
        "2026",
        {"format": "%Y", "grain": "year", "fy_start_month": 10},
        (P(2025, 10, 1), P(2026, 9, 30)),
    ),
    (
        "2025-10-01",
        {"format": "%Y-%m-%d", "grain": "year", "fy_start_month": 10},
        (P(2025, 10, 1), P(2026, 9, 30)),
    ),
    (
        "2025-09-30",
        {"format": "%Y-%m-%d", "grain": "year", "fy_start_month": 10},
        (P(2024, 10, 1), P(2025, 9, 30)),
    ),
    (
        "2025-11-15",
        {"format": "%Y-%m-%d", "grain": "quarter", "fy_start_month": 10},
        (P(2025, 10, 1), P(2025, 12, 31)),
    ),
    (
        "2026-02-15",
        {"format": "%Y-%m-%d", "grain": "quarter", "fy_start_month": 10},
        (P(2026, 1, 1), P(2026, 3, 31)),
    ),
    (
        "2026-02-15",
        {"format": "%Y-%m-%d", "grain": "quarter", "fy_start_month": 2},
        (P(2026, 2, 1), P(2026, 4, 30)),
    ),
    (
        "2026-01-15",
        {"format": "%Y-%m-%d", "grain": "quarter", "fy_start_month": 2},
        (P(2025, 11, 1), P(2026, 1, 31)),
    ),
    (
        "2026-06-30",
        {"format": "%Y-%m-%d", "grain": "year", "fy_start_month": 7},
        (P(2025, 7, 1), P(2026, 6, 30)),
    ),
    (
        "2026-07-01",
        {"format": "%Y-%m-%d", "grain": "year", "fy_start_month": 7},
        (P(2026, 7, 1), P(2027, 6, 30)),
    ),
    (
        "Mar-26",
        {"format": "%b-%y", "grain": "month", "fy_start_month": 10},
        (P(2026, 3, 1), P(2026, 3, 31)),
    ),
    # patterns: fiscal years
    ("FY26", {"pattern": "FY{yy}", "grain": "year"}, (P(2026, 1, 1), P(2026, 12, 31))),
    (
        "FY26",
        {"pattern": "FY{yy}", "grain": "year", "fy_start_month": 10},
        (P(2025, 10, 1), P(2026, 9, 30)),
    ),
    (
        "FY2026",
        {"pattern": "FY{yyyy}", "grain": "year", "fy_start_month": 10},
        (P(2025, 10, 1), P(2026, 9, 30)),
    ),
    (
        "fy26",
        {"pattern": "FY{yy}", "grain": "year", "fy_start_month": 4},
        (P(2025, 4, 1), P(2026, 3, 31)),
    ),
    ("  FY26 ", {"pattern": "FY{yy}", "grain": "year"}, (P(2026, 1, 1), P(2026, 12, 31))),
    # patterns: fiscal quarters
    ("Q1 FY26", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, (P(2026, 1, 1), P(2026, 3, 31))),
    ("Q4 FY26", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, (P(2026, 10, 1), P(2026, 12, 31))),
    (
        "Q1 FY26",
        {"pattern": "Q{q} FY{yy}", "grain": "quarter", "fy_start_month": 10},
        (P(2025, 10, 1), P(2025, 12, 31)),
    ),
    (
        "Q2 FY26",
        {"pattern": "Q{q} FY{yy}", "grain": "quarter", "fy_start_month": 10},
        (P(2026, 1, 1), P(2026, 3, 31)),
    ),
    (
        "Q3 FY26",
        {"pattern": "Q{q} FY{yy}", "grain": "quarter", "fy_start_month": 10},
        (P(2026, 4, 1), P(2026, 6, 30)),
    ),
    (
        "Q4 FY26",
        {"pattern": "Q{q} FY{yy}", "grain": "quarter", "fy_start_month": 10},
        (P(2026, 7, 1), P(2026, 9, 30)),
    ),
    (
        "2026 Q2",
        {"pattern": "{yyyy} Q{q}", "grain": "quarter", "fy_start_month": 4},
        (P(2025, 7, 1), P(2025, 9, 30)),
    ),
    (
        "Q4 FY27",
        {"pattern": "Q{q} FY{yy}", "grain": "quarter", "fy_start_month": 2},
        (P(2026, 11, 1), P(2027, 1, 31)),
    ),
    ("2026-Q3", {"pattern": "{yyyy}-Q{q}", "grain": "quarter"}, (P(2026, 7, 1), P(2026, 9, 30))),
    ("Q2/26", {"pattern": "Q{q}/{yy}", "grain": "quarter"}, (P(2026, 4, 1), P(2026, 6, 30))),
    (
        "Period: FY 2024 (actual)",
        {"pattern": "Period: FY {yyyy} (actual)", "grain": "year"},
        (P(2024, 1, 1), P(2024, 12, 31)),
    ),
    (
        "FY00",
        {"pattern": "FY{yy}", "grain": "year", "fy_start_month": 10},
        (P(1999, 10, 1), P(2000, 9, 30)),
    ),
]

PERIOD_REJECT = [
    ("", {"format": "%b-%y", "grain": "month"}, "period: empty"),
    ("Jan 26", {"format": "%b-%y", "grain": "month"}, "period: unparseable"),
    ("Foo-26", {"format": "%b-%y", "grain": "month"}, "period: unparseable"),
    ("2026-13", {"format": "%Y-%m", "grain": "month"}, "period: unparseable"),
    ("Q5 FY26", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, "period: unparseable"),
    ("Q0 FY26", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, "period: unparseable"),
    ("Q1 FY2026", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, "period: unparseable"),
    ("Q1FY26", {"pattern": "Q{q} FY{yy}", "grain": "quarter"}, "period: unparseable"),
    ("FY26 total", {"pattern": "FY{yy}", "grain": "year"}, "period: unparseable"),
    ("FY6", {"pattern": "FY{yy}", "grain": "year"}, "period: unparseable"),
    ("26", {"pattern": "FY{yy}", "grain": "year"}, "period: unparseable"),
]


@pytest.mark.parametrize(("raw", "params", "want"), PERIOD_OK)
def test_period_ok(raw, params, want):
    assert parse_period(raw, **params) == want
    assert run_ops(raw, [{"op": "period", **params}]) == want


@pytest.mark.parametrize(("raw", "params", "reason"), PERIOD_REJECT)
def test_period_reject(raw, params, reason):
    with pytest.raises(ParseError) as e:
        parse_period(raw, **params)
    assert e.value.reason == reason


@pytest.mark.parametrize(
    "params",
    [
        {"grain": "week", "format": "%Y"},
        {"grain": "month"},
        {"grain": "month", "format": "%b-%y", "pattern": "FY{yy}"},
        {"grain": "year", "pattern": "FY"},
        {"grain": "year", "pattern": "FY{yy} {yyyy}"},
        {"grain": "year", "pattern": "Q{q} FY{yy}"},
        {"grain": "quarter", "pattern": "FY{yy}"},
        {"grain": "quarter", "pattern": "Q{q} Q{q} FY{yy}"},
        {"grain": "year", "pattern": "FY{yy}", "fy_start_month": 0},
        {"grain": "year", "pattern": "FY{yy}", "fy_start_month": 13},
        {"grain": "year", "pattern": "FY{yy}", "fy_start_month": True},
    ],
)
def test_period_bad_params_are_value_errors(params):
    with pytest.raises(ValueError) as e:
        parse_period("FY26", **params)
    assert not isinstance(e.value, ParseError)
    with pytest.raises(ValueError):
        validate_ops([{"op": "period", **params}])


def test_every_month_and_fy_start_partitions_the_year():
    """Fiscal quarters for every fy_start_month tile their fiscal year with no gap or overlap."""
    for fsm in range(1, 13):
        quarters = [
            parse_period(f"Q{q} FY26", pattern="Q{q} FY{yy}", grain="quarter", fy_start_month=fsm)
            for q in range(1, 5)
        ]
        year = parse_period("FY26", pattern="FY{yy}", grain="year", fy_start_month=fsm)
        assert quarters[0][0] == year[0] and quarters[-1][1] == year[1]
        for (_, end), (start, _) in zip(quarters, quarters[1:], strict=False):
            assert (start - end).days == 1
        for q, (start, end) in enumerate(quarters, start=1):
            assert period_of(start, "quarter", fsm) == (start, end), (fsm, q)
            assert period_of(end, "year", fsm) == year


# ---------------------------------------------------------------------- text, regex, map


@pytest.mark.parametrize(
    ("raw", "params", "want"),
    [
        ("  DET-07  ", {}, "DET-07"),
        ("Plant\t\t7", {}, "Plant 7"),
        ("a \n b", {}, "a b"),
        ("  a  b  ", {"collapse": False}, "a  b"),
        ("  a  b  ", {"strip": False}, " a b "),
        ("  a  b  ", {"strip": False, "collapse": False}, "  a  b  "),
        ("p-1234", {"case": "upper"}, "P-1234"),
        ("P-1234", {"case": "lower"}, "p-1234"),
        ("MiXeD", {"case": "none"}, "MiXeD"),
        (" x ", {}, "x"),
        ("", {}, ""),
    ],
)
def test_text(raw, params, want):
    assert apply_text(raw, **params) == want
    assert run_ops(raw, [{"op": "text", **params}]) == want


@pytest.mark.parametrize(
    ("raw", "pattern", "want"),
    [
        ("Period: Mar 2026", r"Period: (.+)", "Mar 2026"),
        ("P-1234", r"P-(\d+)", "1234"),
        ("Site DET-07 (Detroit)", r"^Site (\S+)", "DET-07"),
        ("x Units: 42", r"Units: (\d+)$", "42"),
        ("abc", r"(?i)(ABC)", "abc"),
    ],
)
def test_regex_ok(raw, pattern, want):
    assert apply_regex(raw, pattern) == want
    assert run_ops(raw, [{"op": "regex", "pattern": pattern}]) == want


@pytest.mark.parametrize(
    ("raw", "pattern", "reason"),
    [
        ("xx P-1234", r"P-(\d+)", "regex: no match"),  # fullmatch, not search
        ("P-1234 xx", r"P-(\d+)", "regex: no match"),
        ("Q", r"^Q(\d)?$", "regex: group did not match"),
        ("nothing", r"^Site (\S+)", "regex: no match"),
    ],
)
def test_regex_reject(raw, pattern, reason):
    with pytest.raises(ParseError) as e:
        apply_regex(raw, pattern)
    assert e.value.reason == reason


@pytest.mark.parametrize("pattern", [r"P-\d+", r"(P)-(\d+)", r"(unclosed", r"(?:x)"])
def test_regex_group_count_is_validated(pattern):
    with pytest.raises(ValueError):
        validate_ops([{"op": "regex", "pattern": pattern}])


def test_map():
    values = {"Detroit": "DET-07", " Plant  7 ": "DET-07", "Gary": "GRY-01"}
    assert apply_map("  Detroit ", values) == "DET-07"
    assert apply_map("Plant 7", values) == "DET-07"
    assert apply_map("Plant\t7", values) == "DET-07"
    with pytest.raises(ParseError) as e:
        apply_map("detroit", values)  # exact: case matters
    assert e.value.reason == "map: no mapping"
    assert apply_map("Toledo", values, default="OTHER") == "OTHER"
    assert run_ops("Toledo", [{"op": "map", "values": values, "default": "OTHER"}]) == "OTHER"
    with pytest.raises(ParseError):
        run_ops("Toledo", [{"op": "map", "values": values}])


def test_constant():
    assert constant("anything", value="2026-03") == "2026-03"
    assert run_ops("", [{"op": "constant", "value": "FY26"}]) == "FY26"
    assert run_ops("x", [{"op": "constant", "value": 2026}]) == "2026"


# ------------------------------------------------------------------------------ pipeline


def test_pipeline_regex_then_period():
    ops = [
        {"op": "regex", "pattern": r"Period: (.+)"},
        {"op": "period", "format": "%b %Y", "grain": "month"},
    ]
    assert run_ops("Period: Mar 2026", ops) == (date(2026, 3, 1), date(2026, 3, 31))


def test_pipeline_text_then_number_and_failing_op_is_named():
    ops = [{"op": "text"}, {"op": "regex", "pattern": r"(.*) units"}, {"op": "number"}]
    assert run_ops("  1,234  units ", ops) == Decimal("1234")
    with pytest.raises(ParseError) as e:
        run_ops("1,234 boxes", ops)
    assert e.value.op == "regex"
    with pytest.raises(ParseError) as e:
        run_ops("lots units", ops)
    assert (e.value.op, e.value.reason) == ("number", "number: unparseable")


def test_compiled_pipeline_is_reusable():
    run = compile_ops([{"op": "number", "thousands": "none", "decimal": ","}])
    assert [run(s) for s in ["1,5", "2", "-3,25"]] == [D("1.5"), D("2"), D("-3.25")]


SPEC_ERRORS = [
    [],
    "number",
    [{"thousands": ","}],
    [{"op": "sum"}],
    [{"op": "number", "thousand": ","}],
    [{"op": "number", "thousands": ";"}],
    [{"op": "number", "decimal": "'"}],
    [{"op": "number", "thousands": ",", "decimal": ","}],
    [{"op": "number", "percent": "yes"}],
    [{"op": "text", "case": "title"}],
    [{"op": "text", "strip": 1}],
    [{"op": "regex"}],
    [{"op": "regex", "pattern": 5}],
    [{"op": "map", "values": {}}],
    [{"op": "map", "values": ["a"]}],
    [{"op": "map", "values": {"a": 1}}],
    [{"op": "map", "values": {"a ": "x", "a": "y"}}],
    [{"op": "date"}],
    [{"op": "date", "format": "%Y", "excel_serial": True}],
    [{"op": "period", "format": "%Y"}],
    [{"op": "constant"}],
    [{"op": "constant", "value": True}],
    [{"op": "number"}, {"op": "text"}],
    [{"op": "date", "format": "%Y"}, {"op": "period", "format": "%Y", "grain": "year"}],
    [{"op": "period", "pattern": "FY{yy}", "grain": "year"}, {"op": "number"}],
]


@pytest.mark.parametrize("ops", SPEC_ERRORS, ids=range(len(SPEC_ERRORS)))
def test_spec_errors_are_value_errors_not_parse_errors(ops):
    with pytest.raises(ValueError) as e:
        run_ops("1", ops)
    assert not isinstance(e.value, ParseError)


def test_parse_error_is_not_a_value_error():
    """apply catches ParseError as a reject; a spec bug must not be swallowed the same way."""
    assert not issubclass(ParseError, ValueError)
    with pytest.raises(ParseError):
        run_ops("abc", [{"op": "number"}])


def test_constant_may_follow_a_typed_op():
    ops = [{"op": "number"}, {"op": "constant", "value": "x"}]
    assert run_ops("12", ops) == "x"


@pytest.mark.parametrize("raw", ["DET-07", "ABC 12", "PLT-1007", "XYZ1,000"])
def test_number_does_not_read_codes_as_currency(raw: str) -> None:
    with pytest.raises(ParseError):
        parse_number(raw)


@pytest.mark.parametrize(("raw", "want"), [("EUR 1.234,50", "1234.50"), ("-USD 12", "-12")])
def test_number_still_strips_iso_codes(raw: str, want: str) -> None:
    thousands, decimal = (".", ",") if "," in raw and "." in raw else (",", ".")
    assert parse_number(raw, thousands=thousands, decimal=decimal) == Decimal(want)
