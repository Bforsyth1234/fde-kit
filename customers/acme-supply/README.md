# Acme Supply: a fake customer for testing fde-kit end to end

A made-up HVAC distributor with four branches in Ohio. Every answer in the golden set comes from
a file in this folder, so every check has one known right answer. Findings: [FRICTION.md](FRICTION.md).

| What | Where | Traps planted |
| --- | --- | --- |
| 7 policy docs | `docs/policies/*.md` | boundary ($150,000 is Gold), registered vs unregistered warranty, Dayton closed after 6 pm |
| Conflicting fee | `returns.md` (15%) vs `returns-memo-2026-03.md` (20%) | the memo says the doc is wrong |
| Facts for the docs | `docs/policies/facts.jsonl` | 45-day window in both docs: agreement, not a conflict |
| Messy POS export | `data/branch_sales_2026_q1.csv` | title rows, TOTAL rows, 4 spellings of branch names, `N/A`, values in $000s |
| Wrangle spec | `wrangle/specs/branch_sales.yaml` + `crosswalks/branches.csv` | |
| Custom tool | `acme_tools.py` (`freight_quote`) | 150 miles means zone C |
| Golden set | `evals/make_golden.py` → `evals/golden_acme.jsonl` | 16 cases |

## Run it from scratch

Needs the fde-kit Postgres (`docker compose up -d postgres` in the kit root) and an
`ANTHROPIC_API_KEY` in `.env` for live runs.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e "../..[rag,wrangle]"
```

```bash
.venv/bin/fde rag migrate && .venv/bin/fde rag create-tenant acme --name "Acme Supply"
```

```bash
.venv/bin/fde rag ingest --tenant acme --adapter local_files --dir docs/policies --entity "Acme Supply"
```

```bash
.venv/bin/fde wrangle approve wrangle/specs/branch_sales.yaml --by you && .venv/bin/fde wrangle apply wrangle/specs/branch_sales.yaml data/branch_sales_2026_q1.csv --tenant acme
```

```bash
.venv/bin/fde eval --golden evals/golden_acme.jsonl --tools acme_tools:tools --provider mock
```

```bash
.venv/bin/fde eval --golden evals/golden_acme.jsonl --tools acme_tools:tools --provider anthropic --json evals/live.json
```

Ask one question:

```bash
.venv/bin/fde rag ask --tenant acme --tools acme_tools:tools "What does freight cost on a 200 lb order to zone B?"
```

## Break it on purpose

```bash
.venv/bin/python evals/sabotage.py freight-rate --provider anthropic --ids tool-01,tool-02
```

```bash
.venv/bin/python evals/sabotage.py no-tools --provider anthropic
```

The data bug (dropping `scale: 1000` from the spec) is manual: edit, re-approve, apply, run the
`table-*` cases, then restore the spec and re-apply.
