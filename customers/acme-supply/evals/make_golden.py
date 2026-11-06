# ruff: noqa: E501  (case data: questions and rubrics read better unwrapped)
"""Writes evals/golden_acme.jsonl. Every expected answer comes from a document or the sales CSV
in this project, so each check has one known right answer. Run: python evals/make_golden.py"""

import json
from pathlib import Path

T = "acme"
SALES = "SELECT e.canonical_name, f.field, f.period_start, f.value_num, f.fact_id FROM rag_facts f JOIN rag_entities e USING (entity_id)"


def search(q):
    return {"tool_calls": [{"name": "search_documents", "input": {"query": q, "k": 5}}]}


def sql(where):
    return {"tool_calls": [{"name": "query_table", "input": {"sql": f"{SALES} WHERE {where}"}}]}


def freight(w, z):
    return {"tool_calls": [{"name": "freight_quote", "input": {"weight_lb": w, "zone": z}}]}


def case(id, cat, prompt, checks, mock):
    return {
        "id": id,
        "category": cat,
        "tenant": T,
        "prompt": prompt,
        "checks": checks,
        "mock": mock,
    }


def c(type, value=None):
    return {"type": type, "value": value} if value is not None else {"type": type}


CASES = [
    # --- doc lookups: one document holds the answer --------------------------------------------
    case(
        "doc-01",
        "doc-lookup",
        "How many days does a contractor have to return unused stock?",
        [
            c("contains", "45"),
            c("cites_one_of", ["returns.md", "returns-memo-2026-03.md"]),
            c("cites_only"),
        ],
        [search("return window days"), "Contractors have 45 days from the invoice date [1]."],
    ),
    case(
        "doc-02",
        "doc-lookup",
        "Can a contractor return an unopened cylinder of R-410A?",
        [
            c("cites", "returns.md"),
            c("cites_only"),
            c(
                "judge",
                "The answer says no: refrigerant cylinders cannot be returned, even unopened.",
            ),
        ],
        [
            search("refrigerant cylinder return"),
            "No. Refrigerant cylinders cannot be returned, opened or not [1].",
        ],
    ),
    case(
        "doc-03",
        "doc-lookup",
        "An AcmeAir unit was registered 30 days after install. How long is the compressor covered?",
        [c("regex", r"10[- ]years?"), c("cites", "warranty.md"), c("cites_only")],
        [
            search("compressor warranty registered"),
            "Registered within 60 days, the compressor is covered for 10 years, parts only [1].",
        ],
    ),
    case(
        "doc-04",
        "doc-lookup",
        "The contractor never registered the AcmeAir furnace. What's the heat exchanger warranty?",
        [
            c("regex", r"10[- ]years?"),
            c("cites", "warranty.md"),
            c(
                "judge",
                "The answer gives 10 years for an unregistered heat exchanger and does not say 20 years applies to this unit.",
            ),
        ],
        [
            search("heat exchanger warranty not registered"),
            "Unregistered, the heat exchanger is covered for 10 years, parts [1].",
        ],
    ),
    case(
        "doc-05",
        "doc-lookup",
        "Does the Acme warranty pay for the labor to swap a failed compressor?",
        [c("cites", "warranty.md"), c("judge", "The answer says labor is not covered.")],
        [search("warranty labor"), "No, labor is never covered by the Acme warranty [1]."],
    ),
    case(
        "doc-06",
        "doc-lookup",
        "A contractor spent exactly $150,000 with us last year. What discount do they get this year?",
        [c("contains", "8%"), c("cites", "contractor-tiers.md"), c("cites_only")],
        [
            search("contractor tier discount"),
            "$150,000 or more is Gold, which gets 8% off list [1].",
        ],
    ),
    case(
        "doc-07",
        "doc-lookup",
        "What's the late charge on an overdue net-30 balance?",
        [c("contains", "1.5%"), c("cites", "contractor-tiers.md")],
        [search("late charge overdue"), "Overdue net-30 balances accrue 1.5% per month [1]."],
    ),
    case(
        "doc-08",
        "doc-lookup",
        "A contractor needs a part at 9 pm. Can they pick it up at the Dayton branch?",
        [
            c("cites_one_of", ["emergency-service.md", "branches.md"]),
            c(
                "judge",
                "The answer says Dayton is closed at 9 pm and points to the Columbus 24/7 emergency counter.",
            ),
        ],
        [
            search("after hours emergency counter"),
            "No, Dayton closes at 6:00 pm. Columbus runs a 24/7 emergency counter [1].",
        ],
    ),
    case(
        "doc-09",
        "doc-lookup",
        "Is the Toledo branch open on Saturdays?",
        [c("cites", "branches.md"), c("judge", "The answer says Toledo is not open on Saturday.")],
        [search("Saturday hours"), "No. Only Columbus and Cincinnati open on Saturday [1]."],
    ),
    # --- custom tool ------------------------------------------------------------------------
    case(
        "tool-01",
        "custom-tool",
        "What does freight cost on a 200 lb order to zone B?",
        [c("tool_called", "freight_quote"), c("amount", "$89.00")],
        [freight(200, "B"), "Freight is $89.00."],
    ),
    case(
        "tool-02",
        "custom-tool",
        "How much is freight on a 350 lb order going 150 miles from the branch?",
        [c("tool_called", "freight_quote"), c("amount", "$178.50")],
        [freight(350, "C"), "150 miles is zone C, so freight is $178.50."],
    ),
    # --- conflict ---------------------------------------------------------------------------
    case(
        "conflict-01",
        "conflict",
        "What is the restocking fee on non-defective returns in 2026?",
        # Not `conflict_reported`: the memo resolves this conflict, so the right answer is 20%
        # with the 15% explained, not "the sources disagree" (see FRICTION.md).
        [
            c("contains", "20%"),
            c("cites", "returns.md"),
            c("cites", "returns-memo-2026-03.md"),
            c(
                "judge",
                "The answer says the fee to quote is 20%, and mentions that the policy document still says 15%.",
            ),
        ],
        [
            search("restocking fee"),
            "The policy says 15% [1] but the March memo says 20% [2]; the sources disagree.",
        ],
    ),
    # --- table facts from the wrangled sheet --------------------------------------------------
    case(
        "table-01",
        "table-fact",
        "What were Dayton's net sales in March 2026?",
        [c("tool_called", "query_table"), c("amount", "$260,800"), c("cites_only")],
        [
            sql(
                "e.canonical_name = 'Dayton' AND f.field = 'net_sales_usd' AND f.period_start = '2026-03-01'"
            ),
            "Dayton's net sales in March 2026 were $260,800 [1].",
        ],
    ),
    case(
        "table-02",
        "table-fact",
        "What were Columbus's total net sales for Q1 2026?",
        [c("tool_called", "query_table"), c("amount", "$1,257,000")],
        [
            sql("e.canonical_name = 'Columbus' AND f.field = 'net_sales_usd'"),
            "Columbus: $412,600 + $389,100 + $455,300 = $1,257,000 for Q1 2026 [1][2][3].",
        ],
    ),
    case(
        "table-03",
        "missing-data",
        "What were Cincinnati's net sales in February 2026?",
        [
            c("tool_called", "query_table"),
            c("no_amount", "$0"),
            c(
                "judge",
                "The answer says there is no February 2026 sales figure for Cincinnati and does not give or estimate a number.",
            ),
        ],
        [
            sql("e.canonical_name = 'Cincinnati' AND f.field = 'net_sales_usd'"),
            "There is no February 2026 figure for Cincinnati; only January and March are on file.",
        ],
    ),
    # --- not in the data ----------------------------------------------------------------------
    case(
        "refuse-01",
        "not-on-file",
        "What's the list price of a 3-ton AcmeAir condenser?",
        [
            c("cites_only"),
            c(
                "judge",
                "The answer says the price is not in the available documents and does not invent a price.",
            ),
        ],
        [
            search("3-ton condenser price"),
            "I can't find a price for that unit in Acme's documents.",
        ],
    ),
]

out = Path(__file__).parent / "golden_acme.jsonl"
out.write_text("".join(json.dumps(x) + "\n" for x in CASES))
print(f"wrote {len(CASES)} cases to {out}")
