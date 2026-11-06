# Acme Supply walkthrough: what worked, what didn't

A fake customer (an HVAC distributor, 4 branches) taken from an empty folder to a passing live
eval, then broken three ways to see whether the evals notice. Run on 2026-11-05 against
`claude-sonnet-5-5`, fde-kit at `5aa0606`.

## Results

| Run | Score |
| --- | --- |
| Mock (harness + real tools, scripted model) | 16 / 16 |
| Live, first run | 15 / 16 (the failure was a bad check, see F8) |
| Live, after fixing that check | 16 / 16 |
| Live, after F1–F4 were fixed (plain `fde eval`, no workarounds) | 15 / 16, then 16 / 16 after loosening doc-04 (F14) |

| Sabotage | Mock | Live | Caught? |
| --- | --- | --- | --- |
| Tool bug: zone B freight $0.29/lb instead of $0.27 | 2 / 2 pass | tool-01 fails | **live only** (F7) |
| Prompt edit: "answer from general knowledge, don't call tools" | - | 0 / 16 | yes |
| Data bug: `scale: 1000` dropped from the wrangle spec (sales 1000× too small) | - | 0 / 3 table cases | yes |

## What worked with no help

- **Wrangle on a messy export.** Title rows, a blank row, `TOTAL` subtotals, hand-typed branch
  names (`DAY-02`, `Cincinnati Branch`, `cincinnati`, `TOL-04 `), thousands commas and `N/A`.
  `profile` found the header row and all three TOTAL rows. The dry run gave 22 facts, 0 rejects,
  and `N/A` was skipped, not loaded as 0. Every value matched the CSV.
- **Stale approval gate.** Editing the spec after approval made `apply` refuse and write nothing.
- **Re-export replace.** Re-applying the spec replaced the facts. It did not add to them
  (11 facts before and after, not 22).
- **Conflicts.** `local_files` + `facts.jsonl` made one conflict group for the restocking fee
  (15% vs 20%). The return window (45 days in both documents) did not form one.
- **Live answers.** The agent explained the memo overriding the policy doc. It refused to
  estimate Cincinnati's missing February figure, and noticed the February TOTAL row leaves
  Cincinnati out. No fabricated citations in any run.
- **Citation checks catch lucky guesses.** With tools turned off, the model said "10 years" for
  the compressor warranty from general knowledge, which is correct. `cites warranty.md` still
  failed it.

## Friction, most important first

**F1–F4 are fixed** on branch `feat/kit-as-dependency`: the harness is `fde_kit.evals`, any
tenant in the store is accepted (checked before the first case), and `--tools module:attr` works
on `fde eval`, `fde run`, `fde rag ask` and `fde tools`. This project now uses only those
commands; `evals/run.py` and the monkeypatches are gone.

| # | Problem | Impact | Workaround used here | Fix in fde-kit |
| --- | --- | --- | --- | --- |
| ~~F1~~ fixed | The eval harness (`evals/run_evals.py`) is not in the installed package. `import fde_kit.evals` fails. | A customer project can't run evals without a checkout of the kit repo. | `evals/run.py` puts the checkout on `sys.path`. | Move the harness into `fde_kit.evals` (keep `evals/` as data + a thin shim). |
| ~~F2~~ fixed | `known_tenants()` is hard-coded to the demo and EDGAR tenants. A case with `"tenant": "acme"` is rejected at load. | Can't eval your own tenant at all. | Monkeypatched. | Accept any tenant that exists in the store, or a `--tenant` allowlist flag. |
| ~~F3~~ fixed | A tenant case gets only the three retrieval tools. There's no way to add the customer's own tools. | Custom tools can't be evaluated next to retrieval. | Monkeypatched `case_registry`. | A `--registry module:function` flag (or a `registry` hook) that gets the tenant and returns a registry. |
| ~~F4~~ fixed | `fde rag ask` / `fde run` can't load customer tools. | Every engagement writes its own entry script (`acme_agent.py`). | Wrote one. | Same hook as F3, e.g. `fde rag ask --registry acme_agent:acme_registry`. |
| F7 | Mock mode passed a broken tool. The scripted final answer has "$89.00" in it, so `amount` passes whatever the tool returned. The README says mock mode "tests the harness and the tools". It only shows that the tools return the cited sources. | A tool regression can ship with a green mock CI. | Run live. | Say so in the README; optionally an `amount_in_tool_output` check that also checks the value appears in a tool result. |
| F8 | `conflict_reported` needs a word like "conflict/disagree". When one source resolves the conflict (a memo says the policy doc is wrong), a correct answer fails. | False failure on a correct answer. | Switched that case to `contains 20%` + `cites` both + `judge`. | Document when to use it, or add a `conflict_resolved` check (value X stated, value Y mentioned as superseded). |
| F5 | `fde eval` has no `--golden`. Only `python -m evals.run_evals` does. The report header says `(golden)` for a custom file. | Confusing. | Used the module. | Add `--golden`; show the file name in the header. |
| F6 | A customer folder named `evals/` shadows the kit's `evals` package. | `ModuleNotFoundError: evals.run_evals`. | Path order in `run.py`. | Goes away with F1. |
| F9 | `profile` showed no scale hint for `Net Sales ($000s)`. The doc says `(000s)` is detected; the `$` stops it. | A person could miss the ×1000. The sabotage shows what that costs. | Added `scale: 1000` by hand. | Accept `($000s)`, `($000)`, `(USD 000s)`. |
| F10 | `local_files` can't load a field dictionary. `restocking_fee_pct` etc. reach `query_table` with no definition. | Weaker tool description for doc-derived facts. | None. | Read an optional `dictionary.yaml` beside `facts.jsonl`. |
| F11 | `grain:` beside the column is right for `op: date` and an error for `op: period`. The error message is clear; the two forms are easy to mix up from the examples. | One failed dry run. | Moved it. | Accept both, or show one form in the examples. |
| F12 | Every case needs a `mock` script, even for a set you only ever run live. | Writing 16 mocks took longer than writing 16 checks. | Wrote them (`make_golden.py` helpers). | Allow `"mock": null` for live-only sets, skipped in mock runs. |

| F13 | `tool_called` passes for a tool that isn't registered: the call is counted even though it errors. Found while writing the F3 test. | Same blind spot as F7 in mock mode. | None. | Count only calls that reached a registered tool, or add `tool_ok`. |
| F14 | `contains "10 years"` failed a correct "10-year, parts-only" answer. My check, not the kit. | False failure. | `regex 10[- ]years?`. | None; a reminder that `contains` is for ids and exact strings, and wording goes to `regex` or `judge`. |
| F15 | `pytest` with `FDE_TEST_PG_URL` set to the docker-compose database (what the Quick start suggests) wiped the demo tenants and `acme`: summit/oakline had 0 chunks, acme had 0 facts afterwards. | A dev database silently loses customer data; the rag eval then scored 5/12 until `fde rag seed-demo`. | Re-seeded and reloaded. | Point the tests at a separate database (`fde_test`) in the docs and CI, or have the fixtures refuse a database that has non-test tenants. |

F1–F4 were the same problem: the kit worked as a repo, not yet as a dependency.
