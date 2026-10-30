"""The closed set of parse ops a mapping spec may use (§6 of docs/pr5-wrangle-core.md).

A spec value is a pipeline of ops: the first gets the cell's raw string, each passes its output
on. There are no expressions on purpose: a reviewer (or the PR 7 agent) can read every op, and
every op is strict. A value that doesn't fit what the spec declared raises `ParseError` and
becomes a reject with a cell reference; it is never coerced into something plausible.

Two kinds of failure are kept apart:

- `ParseError` is a **data** failure: this cell doesn't parse. It becomes a reject.
- `ValueError` is a **spec** failure: an unknown op or param, a bad param value, a regex without
  exactly one group, or an op chain that can't type-check. It's raised by `compile_ops` before
  any cell is read, so a broken spec never turns into thousands of rejects.

`ParseError.reason` is a short, stable category ("number: unparseable"), never the raw value:
reports group rejects by reason, and the raw value goes in the reject's own field.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any

Period = tuple[date, date]


class ParseError(Exception):
    """A value didn't parse. `reason` is a stable category; `op` names the failing op when the
    error came through `run_ops`/`compile_ops`."""

    def __init__(self, reason: str, op: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.op = op


# ---------------------------------------------------------------------------------- text


_WS = re.compile(r"\s+")
CASES = ("none", "upper", "lower")


def apply_text(raw: str, *, strip: bool = True, case: str = "none", collapse: bool = True) -> str:
    """Whitespace and case only. `collapse` turns each whitespace run into one space."""
    s = raw
    if collapse:
        s = _WS.sub(" ", s)
    if strip:
        s = s.strip()
    if case == "upper":
        s = s.upper()
    elif case == "lower":
        s = s.lower()
    return s


# --------------------------------------------------------------------------------- regex


def compile_regex(pattern: str) -> re.Pattern[str]:
    """Compile and check: exactly one capture group, or ValueError."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"regex {pattern!r} doesn't compile: {e}") from e
    if rx.groups != 1:
        raise ValueError(f"regex {pattern!r} must have exactly one group, has {rx.groups}")
    return rx


def _anchored(pattern: str) -> bool:
    """Starts with `^`, or ends with a `$` that isn't escaped. A `^` or `$` anywhere else
    (`([^,]+)`, `\\$(\\d+)`) is not an anchor."""
    if pattern.startswith("^"):
        return True
    if not pattern.endswith("$"):
        return False
    backslashes = len(pattern[:-1]) - len(pattern[:-1].rstrip("\\"))
    return backslashes % 2 == 0


def apply_regex(raw: str, pattern: str | re.Pattern[str]) -> str:
    """The group's text. `re.fullmatch`, unless the pattern starts with `^` or ends with an
    unescaped `$`, in which case it's `re.search` and the pattern's own anchors decide."""
    rx = compile_regex(pattern) if isinstance(pattern, str) else pattern
    m = rx.search(raw) if _anchored(rx.pattern) else rx.fullmatch(raw)
    if m is None:
        raise ParseError("regex: no match")
    if m.group(1) is None:
        raise ParseError("regex: group did not match")
    return m.group(1)


# ----------------------------------------------------------------------------------- map


def apply_map(raw: str, values: Mapping[str, str], default: str | None = None) -> str:
    """Exact lookup after `text` defaults (strip, collapse) on both the value and the keys.
    `default=None` means absent: an unmapped value is a reject."""
    table = _normalise_map(values)
    key = apply_text(raw)
    if key in table:
        return table[key]
    if default is not None:
        return default
    raise ParseError("map: no mapping")


def _normalise_map(values: Mapping[str, str]) -> dict[str, str]:
    table: dict[str, str] = {}
    for k, v in values.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"map values must be str -> str, got {k!r}: {v!r}")
        nk = apply_text(k)
        if nk in table and table[nk] != v:
            raise ValueError(f"map keys {k!r} collide after whitespace normalisation")
        table[nk] = v
    return table


# -------------------------------------------------------------------------------- number


THOUSANDS = (",", ".", "'", " ", None)
DECIMALS = (".", ",")
CURRENCY_SYMBOLS = "$€£¥"
# Only real ISO 4217 codes count. Any three capital letters would read a plant code like
# `DET-07` as the "currency" DET and the number -07.
ISO_CURRENCY_CODES = frozenset(
    """AED ARS AUD BGN BRL CAD CHF CLP CNY COP CZK DKK EGP EUR GBP HKD HUF IDR ILS INR ISK JPY
    KES KRW KWD MAD MXN MYR NGN NOK NZD PEN PHP PKR PLN QAR RON RSD RUB SAR SEK SGD THB TRY TWD
    UAH USD VND ZAR""".split()
)
_CURRENCY = (
    rf"(?:[{re.escape(CURRENCY_SYMBOLS)}]|(?:{'|'.join(sorted(ISO_CURRENCY_CODES))})(?![A-Za-z]))"
)
_LEAD_CURRENCY = re.compile(rf"^([+\-−]?)\s*{_CURRENCY}\s*")
_TRAIL_CURRENCY = re.compile(rf"\s*{_CURRENCY}$")
# The characters each declared thousands separator stands for in real exports.
_THOUSANDS_CHARS = {
    ",": ",",
    ".": ".",
    "'": "'’",
    " ": "   ",
}


def _number_body(thousands: str | None, decimal: str) -> re.Pattern[str]:
    if thousands is None:
        integer = r"\d+"
    else:
        t = "[" + re.escape(_THOUSANDS_CHARS[thousands]) + "]"
        integer = rf"(?:\d{{1,3}}(?:{t}\d{{3}})+|\d+)"
    d = re.escape(decimal)
    return re.compile(rf"(?:{integer}(?:{d}\d+)?|{d}\d+)")


def _strip_currency(s: str) -> tuple[str, bool]:
    m = _LEAD_CURRENCY.match(s)
    if m:
        return m.group(1) + s[m.end() :], True
    m = _TRAIL_CURRENCY.search(s)
    if m:
        return s[: m.start()], True
    return s, False


def parse_number(
    raw: str,
    *,
    thousands: str | None = ",",
    decimal: str = ".",
    parens_negative: bool = False,
    trailing_minus: bool = False,
    currency: bool = True,
    percent: bool = False,
) -> Decimal:
    """Strict: after the declared stripping (outer whitespace, currency, parentheses, trailing
    minus, %), what's left must be an optional sign and a number in exactly the declared style.
    Thousands groups must be 3 digits; ungrouped digits are also accepted.

    `percent=True` accepts a trailing `%` and divides by 100 (`12%` -> 0.12); a value without
    `%` is taken as written. `percent=False` rejects any `%`."""
    _check_number_params(thousands, decimal)
    s = raw.strip()
    if not s:
        raise ParseError("number: empty")
    negative = False
    had_currency = False

    if currency:
        s, had_currency = _strip_currency(s)
        s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        if not parens_negative:
            raise ParseError("number: parentheses not allowed")
        negative = True
        s = s[1:-1].strip()
        if currency and not had_currency:
            s, had_currency = _strip_currency(s)
            s = s.strip()
    if s.endswith("-") and len(s) > 1:
        if not trailing_minus:
            raise ParseError("number: unparseable")
        if negative:
            raise ParseError("number: double negative")
        negative = True
        s = s[:-1].rstrip()
        if currency and not had_currency:
            s, had_currency = _strip_currency(s)
            s = s.strip()

    is_percent = False
    if s.endswith("%"):
        if not percent:
            raise ParseError("number: percent not allowed")
        is_percent = True
        s = s[:-1].rstrip()

    if s[:1] in ("-", "+", "−"):
        if s[0] != "+":
            if negative:
                raise ParseError("number: double negative")
            negative = True
        s = s[1:]
        if currency and not had_currency:
            s, had_currency = _strip_currency(s)

    if not _number_body(thousands, decimal).fullmatch(s):
        raise ParseError("number: unparseable")
    plain = s
    if thousands is not None:
        for ch in _THOUSANDS_CHARS[thousands]:
            plain = plain.replace(ch, "")
    if decimal != ".":
        plain = plain.replace(decimal, ".")
    try:
        value = Decimal(plain)
    except InvalidOperation as e:  # pragma: no cover - the regex already guarantees this
        raise ParseError("number: unparseable") from e
    if is_percent:
        value = value.scaleb(-2)
    return -value if negative else value


def _check_number_params(thousands: str | None, decimal: str) -> None:
    if thousands not in THOUSANDS:
        raise ValueError(f"number: thousands must be one of {THOUSANDS}, got {thousands!r}")
    if decimal not in DECIMALS:
        raise ValueError(f"number: decimal must be one of {DECIMALS}, got {decimal!r}")
    if thousands == decimal:
        raise ValueError(f"number: thousands and decimal are both {decimal!r}")


def to_fact_value(d: Decimal) -> int | Decimal:
    """`int` when integral, else the Decimal itself, so `1100.0000000000002` is stored exactly
    (a float would go through Postgres's float8 -> numeric cast, which keeps 15 digits). Never
    rounds."""
    if not d.is_finite():
        raise ValueError(f"not a finite number: {d}")
    if d == d.to_integral_value():
        return int(d)
    return d


# ---------------------------------------------------------------------------------- date


_EXCEL_EPOCH = date(1899, 12, 31)
_EXCEL_MAX_SERIAL = 2958465  # 9999-12-31


def parse_date(raw: str, *, format: str | None = None, excel_serial: bool = False) -> date:
    """`format` (strptime) or `excel_serial` (1900 date system), exactly one.

    Excel serials: 1 is 1900-01-01. Excel counts a 1900-02-29 that never existed (serial 60,
    a Lotus 1-2-3 bug it kept); 60 is a reject and serials above it are shifted back a day. A
    fractional serial (a time of day) keeps only its date."""
    _check_date_params(format, excel_serial)
    s = raw.strip()
    if not s:
        raise ParseError("date: empty")
    if excel_serial:
        try:
            serial = Decimal(s)
        except InvalidOperation as e:
            raise ParseError("date: unparseable") from e
        if not serial.is_finite():
            raise ParseError("date: unparseable")
        days = int(serial)  # truncates the time of day; serials are positive
        if days < 1 or days > _EXCEL_MAX_SERIAL:
            raise ParseError("date: excel serial out of range")
        if days == 60:
            raise ParseError("date: excel serial is 1900-02-29, which doesn't exist")
        return _EXCEL_EPOCH + timedelta(days=days if days < 60 else days - 1)
    assert format is not None
    try:
        return datetime.strptime(s, format).date()
    except ValueError as e:
        raise ParseError("date: unparseable") from e


def _check_date_params(format: str | None, excel_serial: bool) -> None:
    if (format is None) == (not excel_serial):
        raise ValueError("date: give exactly one of `format` or `excel_serial: true`")
    if format is not None and not isinstance(format, str):
        raise ValueError(f"date: format must be a string, got {format!r}")


# -------------------------------------------------------------------------------- period


GRAINS = ("day", "month", "quarter", "year", "instant")
_PATTERN_TOKENS = {"q": r"(?P<q>[1-4])", "yy": r"(?P<yy>\d{2})", "yyyy": r"(?P<yyyy>\d{4})"}
_TOKEN_RE = re.compile(r"\{(q|yy|yyyy)\}")


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + n
    return total // 12, total % 12 + 1


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def fiscal_year_start(fy: int, fy_start_month: int = 1) -> date:
    """FY `fy` is named for the calendar year it ends in: with `fy_start_month=10`, FY2026 is
    2025-10-01..2026-09-30; with 1, it's calendar 2026."""
    return date(fy if fy_start_month == 1 else fy - 1, fy_start_month, 1)


def fiscal_quarter(fy: int, q: int, fy_start_month: int = 1) -> Period:
    start = fiscal_year_start(fy, fy_start_month)
    y, m = _add_months(start.year, start.month, 3 * (q - 1))
    ey, em = _add_months(y, m, 2)
    return date(y, m, 1), _month_end(ey, em)


def period_of(d: date, grain: str, fy_start_month: int = 1) -> Period:
    """The period of `grain` containing `d`. Quarters and years are fiscal ones."""
    _check_period_grain(grain, fy_start_month)
    if grain in ("day", "instant"):
        return d, d
    if grain == "month":
        return date(d.year, d.month, 1), _month_end(d.year, d.month)
    # Months since this date's fiscal year began.
    offset = (d.month - fy_start_month) % 12
    fy_start_year, _ = _add_months(d.year, d.month, -offset)
    if grain == "year":
        ey, em = _add_months(fy_start_year, fy_start_month, 11)
        return date(fy_start_year, fy_start_month, 1), _month_end(ey, em)
    sy, sm = _add_months(fy_start_year, fy_start_month, (offset // 3) * 3)
    ey, em = _add_months(sy, sm, 2)
    return date(sy, sm, 1), _month_end(ey, em)


def compile_period_pattern(pattern: str, grain: str) -> re.Pattern[str]:
    """`Q{q} FY{yy}` -> a regex. Literal text matches exactly (case-insensitive). The pattern
    needs one year token, and `{q}` exactly when the grain is quarter."""
    tokens = _TOKEN_RE.findall(pattern)
    years = [t for t in tokens if t != "q"]
    if len(years) != 1:
        raise ValueError(f"period: pattern {pattern!r} needs exactly one of {{yy}} or {{yyyy}}")
    quarters = tokens.count("q")
    if quarters > 1:
        raise ValueError(f"period: pattern {pattern!r} has more than one {{q}}")
    want = "quarter" if quarters else "year"
    if grain != want:
        raise ValueError(f"period: pattern {pattern!r} gives a {want}, but grain is {grain!r}")
    parts: list[str] = []
    pos = 0
    for m in _TOKEN_RE.finditer(pattern):
        parts.append(re.escape(pattern[pos : m.start()]))
        parts.append(_PATTERN_TOKENS[m.group(1)])
        pos = m.end()
    parts.append(re.escape(pattern[pos:]))
    return re.compile("".join(parts), re.IGNORECASE)


def parse_period(
    raw: str,
    *,
    grain: str,
    format: str | None = None,
    pattern: str | None = None,
    fy_start_month: int = 1,
) -> Period:
    """(start, end), both inclusive.

    - `format` (strptime): the date it gives is expanded to the `grain` period containing it.
      `%b-%y` with grain month -> first..last day of that month. With `%Y` and grain year, the
      year read is the fiscal year (Jan 1 of year Y is always in FY Y).
    - `pattern` with tokens `{q}`, `{yy}` (20yy), `{yyyy}`: fiscal quarter or fiscal year.
    """
    _check_period_params(grain, format, pattern, fy_start_month)
    s = raw.strip()
    if not s:
        raise ParseError("period: empty")
    if format is not None:
        try:
            d = datetime.strptime(s, format).date()
        except ValueError as e:
            raise ParseError("period: unparseable") from e
        return period_of(d, grain, fy_start_month)
    assert pattern is not None
    m = compile_period_pattern(pattern, grain).fullmatch(s)
    if m is None:
        raise ParseError("period: unparseable")
    groups = m.groupdict()
    fy = int(groups["yyyy"]) if groups.get("yyyy") else 2000 + int(groups["yy"])
    if groups.get("q"):
        return fiscal_quarter(fy, int(groups["q"]), fy_start_month)
    start = fiscal_year_start(fy, fy_start_month)
    ey, em = _add_months(start.year, start.month, 11)
    return start, _month_end(ey, em)


def _check_period_grain(grain: str, fy_start_month: int) -> None:
    if grain not in GRAINS:
        raise ValueError(f"period: grain must be one of {GRAINS}, got {grain!r}")
    if (
        not isinstance(fy_start_month, int)
        or isinstance(fy_start_month, bool)
        or not 1 <= fy_start_month <= 12
    ):
        raise ValueError(f"period: fy_start_month must be 1..12, got {fy_start_month!r}")


def _check_period_params(
    grain: str, format: str | None, pattern: str | None, fy_start_month: int
) -> None:
    _check_period_grain(grain, fy_start_month)
    if (format is None) == (pattern is None):
        raise ValueError("period: give exactly one of `format` or `pattern`")
    if pattern is not None:
        compile_period_pattern(pattern, grain)


# ------------------------------------------------------------------------------ constant


def constant(_raw: object, value: str) -> str:
    """Ignores its input; for a fixed period or label."""
    return value


# ------------------------------------------------------------------------------ pipeline

# op -> (params with defaults, `_REQUIRED` marking required ones; input type; output type).
# Types: "str", "any", "decimal", "date", "period".
_REQUIRED = object()
OPS: dict[str, tuple[dict[str, Any], str, str]] = {
    "text": ({"strip": True, "case": "none", "collapse": True}, "str", "str"),
    "regex": ({"pattern": _REQUIRED}, "str", "str"),
    "map": ({"values": _REQUIRED, "default": None}, "str", "str"),
    "number": (
        {
            "thousands": ",",
            "decimal": ".",
            "parens_negative": False,
            "trailing_minus": False,
            "currency": True,
            "percent": False,
        },
        "str",
        "decimal",
    ),
    "date": ({"format": None, "excel_serial": False}, "str", "date"),
    "period": (
        {"format": None, "pattern": None, "grain": _REQUIRED, "fy_start_month": 1},
        "str",
        "period",
    ),
    "constant": ({"value": _REQUIRED}, "any", "str"),
}

Op = Callable[[Any], Any]


def _bool_param(op: str, name: str, v: object) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{op}: {name} must be true or false, got {v!r}")
    return v


def _str_param(op: str, name: str, v: object, *, optional: bool = False) -> str | None:
    if v is None and optional:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{op}: {name} must be a string, got {v!r}")
    return v


def _build(op: str, p: dict[str, Any]) -> Op:
    if op == "text":
        case = p["case"]
        if case is None:
            case = "none"
        if case not in CASES:
            raise ValueError(f"text: case must be one of {CASES}, got {case!r}")
        return partial(
            apply_text,
            strip=_bool_param(op, "strip", p["strip"]),
            case=case,
            collapse=_bool_param(op, "collapse", p["collapse"]),
        )
    if op == "regex":
        return partial(apply_regex, pattern=compile_regex(_str_param(op, "pattern", p["pattern"])))
    if op == "map":
        values = p["values"]
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"map: values must be a non-empty mapping, got {values!r}")
        table = _normalise_map(values)
        default = _str_param(op, "default", p["default"], optional=True)
        return partial(apply_map, values=table, default=default)
    if op == "number":
        thousands = p["thousands"]
        if thousands == "none":
            thousands = None
        _check_number_params(thousands, p["decimal"])
        return partial(
            parse_number,
            thousands=thousands,
            decimal=p["decimal"],
            **{
                k: _bool_param(op, k, p[k])
                for k in ("parens_negative", "trailing_minus", "currency", "percent")
            },
        )
    if op == "date":
        fmt = _str_param(op, "format", p["format"], optional=True)
        excel = _bool_param(op, "excel_serial", p["excel_serial"])
        _check_date_params(fmt, excel)
        return partial(parse_date, format=fmt, excel_serial=excel)
    if op == "period":
        fmt = _str_param(op, "format", p["format"], optional=True)
        pattern = _str_param(op, "pattern", p["pattern"], optional=True)
        grain = _str_param(op, "grain", p["grain"])
        assert grain is not None
        _check_period_params(grain, fmt, pattern, p["fy_start_month"])
        return partial(
            parse_period,
            grain=grain,
            format=fmt,
            pattern=pattern,
            fy_start_month=p["fy_start_month"],
        )
    if op == "constant":
        value = p["value"]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"constant: value must be a string, got {value!r}")
        return partial(constant, value=str(value))
    raise AssertionError(op)  # pragma: no cover


def compile_ops(ops: list[dict[str, Any]]) -> Callable[[str], object]:
    """Validate a pipeline once and return a function of the raw cell string.

    ValueError for an unknown op or param, a bad param, or a chain that doesn't type-check (only
    `constant` may follow an op whose output isn't a string)."""
    if not isinstance(ops, list) or not ops:
        raise ValueError("ops must be a non-empty list")
    steps: list[tuple[str, Op]] = []
    current = "str"
    for i, spec in enumerate(ops):
        if not isinstance(spec, Mapping) or "op" not in spec:
            raise ValueError(f"ops[{i}] must be a mapping with an 'op' key, got {spec!r}")
        name = spec["op"]
        if name not in OPS:
            raise ValueError(f"ops[{i}]: unknown op {name!r}; ops are {sorted(OPS)}")
        defaults, in_type, out_type = OPS[name]
        params = {k: v for k, v in spec.items() if k != "op"}
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            raise ValueError(f"ops[{i}] {name}: unknown params {unknown}; has {sorted(defaults)}")
        merged = {**defaults, **params}
        missing = sorted(k for k, v in merged.items() if v is _REQUIRED)
        if missing:
            raise ValueError(f"ops[{i}] {name}: missing required params {missing}")
        if in_type == "str" and current != "str":
            raise ValueError(f"ops[{i}] {name} needs a string, but the previous op gives {current}")
        try:
            fn = _build(name, merged)
        except ValueError as e:
            raise ValueError(f"ops[{i}] {e}") from e
        steps.append((name, fn))
        current = out_type

    def run(raw: str) -> object:
        value: object = raw
        for name, fn in steps:
            try:
                value = fn(value)
            except ParseError as e:
                e.op = name
                raise
        return value

    return run


def validate_ops(ops: list[dict[str, Any]]) -> None:
    """Raise ValueError if `ops` isn't a valid pipeline (see `compile_ops`)."""
    compile_ops(ops)


def run_ops(raw: str, ops: list[dict[str, Any]]) -> object:
    """Run a pipeline on one raw string. ValueError for a bad pipeline, ParseError for a value
    that doesn't parse."""
    return compile_ops(ops)(raw)
