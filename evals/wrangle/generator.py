"""A seeded, messy, multi-source world with known answers (docs/wrangle-plan.md §5.1).

    python -m evals.wrangle.generator --seed 11 --out /tmp/world

Writes three directories under `--out`:

    sources/     what a customer hands over: four "systems" that were never meant to meet
    reference/   what a mapper may also read: the dictionary, the plant list, the standard chart
    truth/       what a mapper must not read: correct specs and crosswalks, every fact with its
                 cell, the injected conflicts, the cells that really can't load, the mess log

Every injected mess has an id in `truth/messes.jsonl`, and each truth fact lists the ids that
touch it, so scores can be broken down by mess type. Same seed, same CSVs, same truth files and
same XLSX cells (an XLSX's bytes may differ: openpyxl stamps the zip entries with the time).
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

N_PLANTS = 20
PARTS_PER_PLANT = 15  # 20 x 15 = the plan's ~300 parts
FIRST_MONTH = date(2024, 1, 1)
N_MONTHS = 24
TB_PERIOD = date(2025, 12, 1)
C_SHEET = "legacy_report"
"""A CSV's sheet name in apply's cells and rejects: the file's stem."""
CLOSED_PLANT_CODE = "GRR-03"
"""A real plant in the legacy report that closed: in no crosswalk, so its rows must be rejects."""

SOURCE_FILES = {
    "A": "erp_extract.csv",
    "B": "planner_forecast.xlsx",
    "C": "legacy_report.csv",
    "D": "trial_balances.xlsx",
}

NA_SITES = [
    ("Detroit", "DET"),
    ("Toledo", "TOL"),
    ("Akron", "AKR"),
    ("Dayton", "DAY"),
    ("Saltillo", "SAL"),
    ("Windsor", "WIN"),
    ("Memphis", "MEM"),
]
EU_SITES = [("Lyon", "LYO"), ("Turin", "TUR"), ("Gdansk", "GDA"), ("Stuttgart", "STU")]
FUNCTIONS = ["Stamping", "Assembly", "Castings", "Machining"]

STANDARD_COA = [
    "1000 Cash",
    "1100 Accounts Receivable",
    "1200 Inventory",
    "1500 Fixed Assets",
    "2000 Accounts Payable",
    "2100 Accrued Liabilities",
    "3000 Retained Earnings",
    "4000 Net Revenue",
    "5000 Cost of Goods Sold",
    "6000 Selling Expense",
    "6100 Administrative Expense",
    "6200 Depreciation Expense",
]
# Three local charts, in STANDARD_COA order: (code format, account names).
LOCAL_CHARTS = {
    "X": (
        "{:05d}",
        [
            "Cash and Bank",
            "Trade Receivables",
            "Stock on Hand",
            "PP&E Net",
            "Trade Payables",
            "Accruals",
            "Retained Profit",
            "Sales",
            "Cost of Sales",
            "Selling Costs",
            "G&A",
            "Depreciation",
        ],
    ),
    "Y": (
        "GL-{:04d}",
        [
            "Petty Cash & Banks",
            "A/R Trade",
            "Inventories",
            "Fixed Assets - Net",
            "A/P",
            "Accrued Liabilities",
            "Retained Earnings",
            "Net Revenue",
            "COGS",
            "Sales Expense",
            "Administration",
            "Depreciation Expense",
        ],
    ),
    "Z": (
        "K{:03d}",
        [
            "Bank",
            "Debtors",
            "Stocks",
            "Plant & Machinery",
            "Creditors",
            "Accrued Expenses",
            "Reserves",
            "Turnover",
            "Direct Costs",
            "Distribution Costs",
            "Overheads",
            "Amortisation",
        ],
    ),
}

DICTIONARY = [
    ("shipped_units", "Units shipped from the plant to customers in the period.", "units"),
    ("returned_units", "Units returned by customers, as a negative number.", "units"),
    ("revenue_usd", "Net revenue for the plant in the period, in US dollars.", "USD"),
    ("cogs_usd", "Cost of goods sold for the plant in the period, in US dollars.", "USD"),
    ("onhand_units", "On-hand inventory at the end of the period, in units.", "units"),
    ("demand_forecast_units", "Forecast customer demand for the period, in units.", "units"),
    ("gl_balance_usd", "Closing balance of a general ledger account, in US dollars.", "USD"),
]
_FIXED_TIME = datetime(2026, 1, 1)
HEADER_VARIANTS = {
    "A_shipped": ["QTY_SHIPPED", "SHIP_QTY", "DESP_QTY"],
    "A_revenue": ["NET_REV", "SLS_AMT", "TURNOVER_AMT"],
    "A_cogs": ["COGS_AMT", "STD_COST_AMT", "CGS"],
    "A_onhand": ["ONHAND_QTY", "STK_QTY", "QOH"],
    "C_shipped": ["Shipped (000s)", "Despatched (000s)", "Units Out (000s)"],
    "C_returns": ["Returns", "Credits", "RMA Qty"],
    "C_revenue": ["Revenue", "Sales Value", "Turnover"],
    "D_balance": ["Balance", "Closing", "YTD Amt"],
}
"""Each seed's systems name their columns differently, so column mapping is measured on labels
the baseline's synonyms weren't all written for."""
UNPIVOT = "*unpivot*"
"""`truth/columns.jsonl`'s column name for a measure read from unpivoted cells."""


@dataclass
class Plant:
    number: int
    city: str
    function: str
    region: str  # "NA" | "EU"
    chart: str
    canonical: str
    erp_code: str
    legacy_code: str
    planner_name: str
    typo_name: str
    parts: list[str] = field(default_factory=list)


@dataclass
class Truth:
    """What the renderers record while they write the sources."""

    facts: list[dict] = field(default_factory=list)
    messes: list[dict] = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    seen_entities: set[tuple[str, str]] = field(default_factory=set)
    mess_ids: set[str] = field(default_factory=set)

    def mess(self, mess_type: str, source: str, detail: str = "", key: str = "") -> str:
        """Log one injected mess; a one-source mess has the id `B-scale_000s`, a repeated one
        adds what it touched (`A-duplicate_row-0007`)."""
        mess_id = f"{source}-{mess_type}" + (f"-{key}" if key else "")
        if mess_id not in self.mess_ids:
            self.mess_ids.add(mess_id)
            self.messes.append(
                {"id": mess_id, "type": mess_type, "source": source, "detail": detail}
            )
        return mess_id

    def fact(self, source: str, entity: str, fld: str, month: date, value, cell: str, **kw) -> None:
        self.facts.append(
            {
                "source": source,
                "entity": entity,
                "field": fld,
                "period_start": month.isoformat(),
                "period_end": month_end(month).isoformat(),
                "value": str(value),
                "precision": kw.get("precision"),
                "cell": cell,
                "messes": kw.get("messes", []),
            }
        )

    def entity(self, source: str, raw: str, canonical: str | None) -> None:
        if (source, raw) not in self.seen_entities:
            self.seen_entities.add((source, raw))
            self.entities.append({"source": source, "raw": raw, "canonical": canonical})


def month_end(month: date) -> date:
    """Every fact here has month grain: it runs from the 1st to the last day, as apply stores it."""
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def _months() -> list[date]:
    y, m = FIRST_MONTH.year, FIRST_MONTH.month
    out = []
    for _ in range(N_MONTHS):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _col(n: int) -> str:
    """1 -> A, 27 -> AA."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _typo(rng: random.Random, name: str) -> str:
    """Swap two adjacent letters inside the city name; never the same string back."""
    word = name.split()[0]
    spots = [i for i in range(1, len(word) - 1) if word[i] != word[i + 1]]
    i = rng.choice(spots)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :] + name[len(word) :]


def build_plants(rng: random.Random) -> list[Plant]:
    """20 plants, with cities that repeat, so `Detroit Stamping 07` has near neighbours like
    `Detroit Stamping 17`. Three codes in the ERP are irregular: their last two digits are
    another plant's number, the trap for anyone reading a plant out of the code. Two legacy
    codes have an unrelated prefix."""
    numbers = rng.sample(range(1, 40), N_PLANTS)
    sites = NA_SITES * 2 + EU_SITES
    plants = []
    for i, number in enumerate(numbers):
        (city, prefix), region = sites[i % len(sites)], "NA"
        if (city, prefix) in EU_SITES:
            region = "EU"
        function = FUNCTIONS[0] if i < 2 else rng.choice(FUNCTIONS)
        canonical = f"{city} {function} {number:02d}"
        plants.append(
            Plant(
                number=number,
                city=city,
                function=function,
                region=region,
                chart="Z" if region == "EU" else ("X" if i % 2 else "Y"),
                canonical=canonical,
                erp_code=str(1000 + number),
                legacy_code=f"{prefix}-{number:02d}",
                planner_name=f"{city} {function} #{number}",
                typo_name=_typo(rng, canonical),
            )
        )
    picked = rng.sample(range(N_PLANTS), 3)
    for a, b in zip(picked, picked[1:] + picked[:1], strict=True):
        plants[a].erp_code = f"{rng.randint(2, 8)}{plants[b].number + 100:03d}"
    for i in rng.sample(range(N_PLANTS), 2):
        plants[i].legacy_code = f"ZX{i}-{plants[i].number:02d}"
    part_ids = rng.sample(range(10000, 99999), N_PLANTS * PARTS_PER_PLANT)
    for i, p in enumerate(plants):
        chunk = part_ids[i * PARTS_PER_PLANT : (i + 1) * PARTS_PER_PLANT]
        p.parts = [f"P-{n}" for n in sorted(chunk)]
    return plants


def pick_headers(rng: random.Random) -> dict[str, str]:
    return {key: rng.choice(options) for key, options in HEADER_VARIANTS.items()}


def _world_numbers(rng: random.Random, plants: list[Plant]) -> dict:
    out = {}
    for p in plants:
        base = rng.randint(8, 70) * 1000
        for m in _months():
            shipped = int(base * rng.uniform(0.8, 1.2))
            revenue = shipped * rng.randint(40, 90)
            out[p.canonical, m] = {
                "shipped": shipped,
                "revenue": revenue,
                "cogs": int(revenue * rng.uniform(0.55, 0.8)),
                "onhand": int(shipped * rng.uniform(0.2, 0.9)),
                "returns": rng.choice([0, 0, rng.randint(20, 4000)]),
            }
    return out


def _write_csv(path: Path, rows: list[list[str]], delimiter: str = ",") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=delimiter, lineterminator="\n").writerows(rows)


def render_a(
    rng: random.Random, out: Path, plants: list[Plant], world: dict, t: Truth, labels: dict
) -> None:
    """ERP extract: long, one row per plant-month. Numeric plant codes, YYYYMMDD month-end
    posting dates, trailing spaces, exact duplicate rows."""
    dates = t.mess("date_yyyymmdd", "A", "POST_DT is YYYYMMDD")
    rows = [
        [
            "PLANT_CD",
            "POST_DT",
            labels["A_shipped"],
            labels["A_revenue"],
            labels["A_cogs"],
            labels["A_onhand"],
        ]
    ]
    columns = [
        ("shipped_units", "shipped", "C"),
        ("revenue_usd", "revenue", "D"),
        ("cogs_usd", "cogs", "E"),
        ("onhand_units", "onhand", "F"),
    ]
    for p in plants:
        irregular = p.erp_code != str(1000 + p.number)
        code = t.mess(
            "irregular_plant_code" if irregular else "numeric_plant_code",
            "A",
            f"{p.erp_code} is {p.canonical}",
            p.erp_code,
        )
        t.entity("A", p.erp_code, p.canonical)
        for m in _months():
            n = world[p.canonical, m]
            row_no = len(rows) + 1
            last_day = calendar.monthrange(m.year, m.month)[1]
            ids = [dates, code]
            plant_cell = p.erp_code
            qty_cell = str(n["shipped"])
            if rng.random() < 0.05:
                ids.append(t.mess("trailing_spaces", "A", f"row {row_no}", f"{row_no:04d}"))
                plant_cell += "  "
                qty_cell += " "
            row = [
                plant_cell,
                f"{m.year}{m.month:02d}{last_day:02d}",
                qty_cell,
                str(n["revenue"]),
                str(n["cogs"]),
                str(n["onhand"]),
            ]
            rows.append(row)
            if rng.random() < 0.02:
                dup_row = row_no + 1
                ids.append(t.mess("duplicate_row", "A", f"row {dup_row}", f"{dup_row:04d}"))
                rows.append(list(row))
            for fld, key, letter in columns:
                t.fact(
                    "A",
                    p.canonical,
                    fld,
                    m,
                    n[key],
                    f"erp_extract!{letter}{row_no}",
                    messes=ids,
                )
    _write_csv(out / SOURCE_FILES["A"], rows)


def render_b(rng: random.Random, out: Path, plants: list[Plant], t: Truth) -> None:
    """Planner workbook: wide, months across. Title rows, a merged two-row header, `(000s)`,
    a Subtotal row per plant and a Total at the end, names like `Detroit Stamping #7`."""
    import openpyxl

    source_messes = [
        t.mess("title_rows", "B", "rows 1-2 are titles"),
        t.mess("merged_header", "B", "`Forecast` merged over the month columns"),
        t.mess("scale_000s", "B", "units are in thousands, said only in a title row"),
        t.mess("subtotal_rows", "B", "a Subtotal after each plant, a Total at the end"),
        t.mess("wide_layout", "B", "months across the columns"),
    ]
    wb = openpyxl.Workbook()
    wb.properties.created = wb.properties.modified = _FIXED_TIME
    ws = wb.active
    ws.title = "Plan"
    ws["A1"] = "Demand Plan"
    ws["A2"] = "Units (000s)"
    months = _months()
    first = 3
    ws.cell(4, first, "Forecast")
    ws.merge_cells(start_row=4, start_column=first, end_row=4, end_column=first + N_MONTHS - 1)
    ws.cell(5, 1, "Site")
    ws.cell(5, 2, "Part No")
    for i, m in enumerate(months):
        ws.cell(5, first + i, m.strftime("%b-%y"))
    ws.cell(5, first + N_MONTHS, "Total")
    row = 6
    grand = [Decimal(0)] * N_MONTHS
    for p in plants:
        name = t.mess("plant_name_variant", "B", f"{p.planner_name} is {p.canonical}", p.number)
        t.entity("B", p.planner_name, p.canonical)
        subtotal = [Decimal(0)] * N_MONTHS
        for part in p.parts:
            ws.cell(row, 1, p.planner_name)
            ws.cell(row, 2, part)
            base = rng.randint(5, 200)
            total = Decimal(0)
            for i, m in enumerate(months):
                if rng.random() < 0.01:  # no forecast for this month: an empty cell, no fact
                    continue
                thousands = Decimal(max(1, int(base * rng.uniform(0.7, 1.3)))) / 10
                ws.cell(row, first + i, float(thousands))
                subtotal[i] += thousands
                total += thousands
                t.fact(
                    "B",
                    f"{p.canonical}/{part}",
                    "demand_forecast_units",
                    m,
                    int(thousands * 1000),
                    f"Plan!{_col(first + i)}{row}",
                    precision=100,  # one decimal of thousands
                    messes=[*source_messes, name],
                )
            ws.cell(row, first + N_MONTHS, float(total))
            row += 1
        ws.cell(row, 1, "Subtotal")
        for i, v in enumerate(subtotal):
            ws.cell(row, first + i, float(v))
            grand[i] += v
        row += 1
    ws.cell(row, 1, "Total")
    for i, v in enumerate(grand):
        ws.cell(row, first + i, float(v))
    wb.save(out / SOURCE_FILES["B"])


def _money(value: Decimal | int, region: str, decimals: int = 0) -> str:
    text = f"{Decimal(value):,.{decimals}f}"
    if region == "EU":
        return text.translate(str.maketrans(",.", ".,"))
    return text


def render_c(
    rng: random.Random, out: Path, plants: list[Plant], world: dict, t: Truth, labels: dict
) -> None:
    """Legacy report: long, plant-month, `;` delimited. `DET-07` codes with some typed names,
    DD/MM/YYYY with a random day (so some are ambiguous on their own), parentheses for returns,
    European numbers for the EU region only, shipped in rounded thousands, and values that
    disagree with the ERP."""
    euro = t.mess("european_decimals", "C", "EU rows write 1.234,56, NA rows 1,234.56")
    rounded = t.mess("rounded_thousands", "C", "Shipped is in whole thousands: agrees within 1000")
    rows = [
        [
            "Region",
            "Plant",
            "Report Date",
            labels["C_shipped"],
            labels["C_returns"],
            labels["C_revenue"],
        ]
    ]
    for p in plants:
        t.entity("C", p.legacy_code, p.canonical)
        code = t.mess("legacy_code", "C", f"{p.legacy_code} is {p.canonical}", p.number)
        for m in _months():
            n = world[p.canonical, m]
            row_no = len(rows) + 1
            base_ids = [code] + ([euro] if p.region == "EU" else [])
            plant_cell = p.legacy_code
            if rng.random() < 0.03:
                plant_cell = p.typo_name
                t.entity("C", p.typo_name, p.canonical)
                base_ids.append(t.mess("typo_plant_name", "C", p.typo_name, f"{row_no:04d}"))
            day = rng.randint(1, 28)
            if day <= 12 and day != m.month:  # 05/05 reads the same either way round
                base_ids.append(t.mess("ambiguous_date", "C", f"row {row_no}", f"{row_no:04d}"))
            shipped_k = int((Decimal(n["shipped"]) / 1000).quantize(Decimal(1), ROUND_HALF_UP))
            revenue = n["revenue"]
            if rng.random() < 0.03:  # a disagreement well outside the 1000 precision
                shipped_k += rng.choice([-1, 1]) * rng.randint(3, 6)
                mess = t.mess("value_conflict", "C", f"row {row_no} shipped", f"{row_no:04d}s")
                t.conflicts.append(
                    {
                        "entity": p.canonical,
                        "field": "shipped_units",
                        "period": m.isoformat(),
                        "period_end": month_end(m).isoformat(),
                    }
                )
                shipped_ids = [*base_ids, mess]
            else:
                shipped_ids = base_ids
            if rng.random() < 0.03:
                revenue = int(revenue * rng.uniform(1.02, 1.1))
                mess = t.mess("value_conflict", "C", f"row {row_no} revenue", f"{row_no:04d}r")
                t.conflicts.append(
                    {
                        "entity": p.canonical,
                        "field": "revenue_usd",
                        "period": m.isoformat(),
                        "period_end": month_end(m).isoformat(),
                    }
                )
                revenue_ids = [*base_ids, mess]
            else:
                revenue_ids = base_ids
            returns = n["returns"]
            returns_cell = f"({_money(returns, p.region)})" if returns else "0"
            returns_ids = base_ids
            if returns:
                returns_ids = [*base_ids, t.mess("parens_negative", "C", "returns in parentheses")]
            if rng.random() < 0.004:
                returns_cell = "n/a"
                t.rejects.append(
                    {"source": "C", "sheet": C_SHEET, "cell": f"E{row_no}", "why": "not a number"}
                )
            else:
                t.fact(
                    "C",
                    p.canonical,
                    "returned_units",
                    m,
                    -returns,
                    f"legacy_report!E{row_no}",
                    messes=returns_ids,
                )
            rows.append(
                [
                    p.region,
                    plant_cell,
                    f"{day:02d}/{m.month:02d}/{m.year}",
                    str(shipped_k),
                    returns_cell,
                    _money(revenue, p.region, 2),
                ]
            )
            t.fact(
                "C",
                p.canonical,
                "shipped_units",
                m,
                shipped_k * 1000,
                f"legacy_report!D{row_no}",
                precision=1000,
                messes=[*shipped_ids, rounded],
            )
            t.fact(
                "C",
                p.canonical,
                "revenue_usd",
                m,
                revenue,
                f"legacy_report!F{row_no}",
                messes=revenue_ids,
            )
    closed = t.mess("closed_plant", "C", f"{CLOSED_PLANT_CODE} is in no crosswalk")
    t.entity("C", CLOSED_PLANT_CODE, None)
    for m in _months()[:4]:
        row_no = len(rows) + 1
        rows.append(
            ["NA", CLOSED_PLANT_CODE, f"28/{m.month:02d}/{m.year}", "9", "0", "1,000.00"],
        )
        t.rejects.append(
            {
                "source": "C",
                "sheet": C_SHEET,
                "row": row_no,
                "why": "unknown entity",
                "mess": closed,
            }
        )
    _write_csv(out / SOURCE_FILES["C"], rows, delimiter=";")


def render_d(
    rng: random.Random, out: Path, plants: list[Plant], t: Truth, labels: dict
) -> dict[str, str]:
    """Trial balances: one sheet per plant, on one of three local charts of accounts. Mapping
    the local codes onto the standard chart is a *value* crosswalk. Returns code -> standard."""
    import openpyxl

    wb = openpyxl.Workbook()
    wb.properties.created = wb.properties.modified = _FIXED_TIME
    wb.remove(wb.active)
    coa_map: dict[str, str] = {}
    mess = t.mess("local_chart", "D", "three local charts onto one standard chart")
    label = TB_PERIOD.strftime("%b %Y")
    for p in plants:
        ws = wb.create_sheet(f"TB {p.number:02d}")
        ws["A1"] = f"Trial balance: {p.planner_name}"
        ws["A2"] = f"Period: {label}"
        for c, text in enumerate(
            ["Plant", "Local Acct", "Account Name", labels["D_balance"]], start=1
        ):
            ws.cell(4, c, text)
        t.entity("D", p.planner_name, p.canonical)
        fmt, names = LOCAL_CHARTS[p.chart]
        for i, standard in enumerate(STANDARD_COA):
            row = 5 + i
            code = fmt.format(100 + i * 10)
            coa_map[code] = standard
            balance = rng.randint(-5_000_000, 5_000_000)
            ws.cell(row, 1, p.planner_name)
            ws.cell(row, 2, code)
            ws.cell(row, 3, names[i])
            ws.cell(row, 4, balance)
            t.fact(
                "D",
                f"{p.canonical}/{standard}",
                "gl_balance_usd",
                TB_PERIOD,
                balance,
                f"{ws.title}!D{row}",
                messes=[mess],
            )
    wb.save(out / SOURCE_FILES["D"])
    return coa_map


_PLANT_PART = {"crosswalk": "plants", "policy": "closed"}


def _legacy_spec(
    name: str, skip_region: str, thousands: str, decimal: str, note: str, labels: dict
) -> dict:
    number = {"op": "number", "thousands": thousands, "decimal": decimal}
    return {
        "spec": name,
        "version": 1,
        "dictionary": "../dictionary.yaml",
        "crosswalks": {"plants": {"path": "../crosswalks/plants.csv", "kind": "plant"}},
        "source": {
            "header_rows": [1],
            "delimiter": ";",
            "skip_rows": [{"column": "Region", "matches": f"^{skip_region}$"}],
        },
        "entity": {
            "kind": "plant",
            "parts": [
                {
                    "column": "Plant",
                    "ops": [{"op": "text", "strip": True}],
                    "resolve": dict(_PLANT_PART),
                }
            ],
        },
        "period": {
            "column": {
                "name": "Report Date",
                "ops": [{"op": "date", "format": "%d/%m/%Y"}],
                "grain": "month",
            }
        },
        "measures": [
            {
                "field": "shipped_units",
                "column": labels["C_shipped"],
                "ops": [number],
                "scale": 1000,
                "precision": 1000,
                "unit": "units",
            },
            {
                "field": "returned_units",
                "column": labels["C_returns"],
                "ops": [{**number, "parens_negative": True}],
                "unit": "units",
            },
            {"field": "revenue_usd", "column": labels["C_revenue"], "ops": [number], "unit": "USD"},
        ],
        "notes": note,
    }


def truth_specs(labels: dict) -> dict[str, dict]:
    number = {"op": "number", "thousands": ",", "decimal": "."}
    return {
        "erp_extract": {
            "spec": "erp_extract",
            "version": 1,
            "dictionary": "../dictionary.yaml",
            "crosswalks": {"plants": {"path": "../crosswalks/plants.csv", "kind": "plant"}},
            "source": {"header_rows": [1], "delimiter": ","},
            "entity": {
                "kind": "plant",
                "parts": [
                    {
                        "column": "PLANT_CD",
                        "ops": [{"op": "text", "strip": True}],
                        "resolve": dict(_PLANT_PART),
                    }
                ],
            },
            "period": {
                "column": {
                    "name": "POST_DT",
                    "ops": [{"op": "date", "format": "%Y%m%d"}],
                    "grain": "month",
                }
            },
            "measures": [
                {
                    "field": fld,
                    "column": col,
                    "ops": [{"op": "text", "strip": True}, number],
                    "unit": unit,
                }
                for fld, col, unit in [
                    ("shipped_units", labels["A_shipped"], "units"),
                    ("revenue_usd", labels["A_revenue"], "USD"),
                    ("cogs_usd", labels["A_cogs"], "USD"),
                    ("onhand_units", labels["A_onhand"], "units"),
                ]
            ],
        },
        "planner_forecast": {
            "spec": "planner_forecast",
            "version": 1,
            "dictionary": "../dictionary.yaml",
            "crosswalks": {"plants": {"path": "../crosswalks/plants.csv", "kind": "plant"}},
            "source": {
                "sheet": "Plan",
                "header_rows": [4, 5],
                "data_starts": 6,
                "skip_rows": [{"column": "Site", "matches": "(?i)^(sub)?total"}],
            },
            "entity": {
                "kind": "plant_part",
                "parts": [
                    {
                        "column": "Site",
                        "ops": [{"op": "text", "strip": True}],
                        "resolve": dict(_PLANT_PART),
                    },
                    {
                        "column": "Part No",
                        "ops": [{"op": "text", "strip": True, "case": "upper"}],
                        "resolve": {"policy": "open"},
                    },
                ],
                "template": "{0}/{1}",
            },
            "period": {
                "unpivot": {
                    "matching": r"^Forecast / [A-Z][a-z]{2}-\d{2}$",
                    "ops": [
                        {"op": "regex", "pattern": "Forecast / (.+)"},
                        {"op": "period", "format": "%b-%y", "grain": "month"},
                    ],
                }
            },
            "measures": [
                {
                    "field": "demand_forecast_units",
                    "ops": [number],
                    "scale": 1000,
                    "precision": 100,
                    "unit": "units",
                }
            ],
        },
        "legacy_report_na": _legacy_spec(
            "legacy_report_na",
            "EU",
            ",",
            ".",
            "NA rows: 1,234.56. EU rows have their own spec.",
            labels,
        ),
        "legacy_report_eu": _legacy_spec(
            "legacy_report_eu",
            "NA",
            ".",
            ",",
            "EU rows: 1.234,56. NA rows have their own spec.",
            labels,
        ),
        "trial_balances": {
            "spec": "trial_balances",
            "version": 1,
            "dictionary": "../dictionary.yaml",
            "crosswalks": {
                "plants": {"path": "../crosswalks/plants.csv", "kind": "plant"},
                "coa": {"path": "../crosswalks/coa.csv", "kind": "account"},
            },
            "source": {
                "sheets": {"matching": "^TB "},
                "header_rows": [4],
            },
            "entity": {
                "kind": "plant_account",
                "parts": [
                    {
                        "column": "Plant",
                        "ops": [{"op": "text", "strip": True}],
                        "resolve": dict(_PLANT_PART),
                    },
                    {
                        "column": "Local Acct",
                        "ops": [{"op": "text", "strip": True}],
                        "resolve": {"crosswalk": "coa", "policy": "closed"},
                    },
                ],
                "template": "{0}/{1}",
            },
            "period": {
                "cell": {
                    "ref": "A2",
                    "ops": [
                        {"op": "regex", "pattern": "Period: (.+)"},
                        {"op": "period", "format": "%b %Y", "grain": "month"},
                    ],
                }
            },
            "measures": [
                {
                    "field": "gl_balance_usd",
                    "column": labels["D_balance"],
                    "ops": [{"op": "number"}],
                }
            ],
        },
    }


SPEC_SOURCE = {
    "erp_extract": "A",
    "planner_forecast": "B",
    "legacy_report_na": "C",
    "legacy_report_eu": "C",
    "trial_balances": "D",
}


def truth_columns(labels: dict) -> list[dict]:
    """source column -> dictionary field, with the scale and unit a right mapping declares."""
    rows = []
    for name, spec in truth_specs(labels).items():
        for m in spec["measures"]:
            rows.append(
                {
                    "source": SPEC_SOURCE[name],
                    "column": m.get("column") or UNPIVOT,
                    "field": m["field"],
                    "scale": m.get("scale", 1),
                    "unit": m.get("unit") or {f: u for f, _, u in DICTIONARY}[m["field"]],
                }
            )
    unique = {json.dumps(r, sort_keys=True): r for r in rows}
    return sorted(unique.values(), key=lambda r: (r["source"], r["column"]))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def generate(seed: int, out: str | Path) -> Path:
    """Build the world for `seed` under `out` and return `out`."""
    import yaml

    rng = random.Random(seed)
    out = Path(out)
    sources, reference, truth = out / "sources", out / "reference", out / "truth"
    project = truth / "project"
    for d in (sources, reference, project / "specs", project / "crosswalks"):
        d.mkdir(parents=True, exist_ok=True)

    plants = build_plants(rng)
    world = _world_numbers(rng, plants)
    t = Truth()
    labels = pick_headers(rng)
    render_a(rng, sources, plants, world, t, labels)
    render_b(rng, sources, plants, t)
    render_c(rng, sources, plants, world, t, labels)
    coa_map = render_d(rng, sources, plants, t, labels)

    dictionary = yaml.safe_dump(
        [{"field": f, "definition": d, "unit": u} for f, d, u in DICTIONARY], sort_keys=False
    )
    (reference / "dictionary.yaml").write_text(dictionary, encoding="utf-8")
    (project / "dictionary.yaml").write_text(dictionary, encoding="utf-8")
    _write_csv(reference / "plants.csv", [["canonical"], *[[p.canonical] for p in plants]])
    _write_csv(reference / "standard_coa.csv", [["canonical"], *[[c] for c in STANDARD_COA]])

    aliases = sorted(
        {
            (e["raw"], e["canonical"])
            for e in t.entities
            if e["canonical"] is not None and e["raw"] != e["canonical"]
        }
    )
    _write_csv(project / "crosswalks" / "plants.csv", [["alias", "canonical"], *aliases])
    _write_csv(
        project / "crosswalks" / "coa.csv",
        [["alias", "canonical"], *sorted(coa_map.items())],
    )
    for name, spec in truth_specs(labels).items():
        (project / "specs" / f"{name}.yaml").write_text(
            yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
        )

    _write_jsonl(truth / "facts.jsonl", t.facts)
    _write_jsonl(truth / "messes.jsonl", t.messes)
    _write_jsonl(truth / "rejects.jsonl", t.rejects)
    _write_jsonl(truth / "conflicts.jsonl", t.conflicts)
    _write_jsonl(truth / "entities.jsonl", t.entities)
    _write_jsonl(truth / "columns.jsonl", truth_columns(labels))
    (truth / "coa.json").write_text(json.dumps(coa_map, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(generate(args.seed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
