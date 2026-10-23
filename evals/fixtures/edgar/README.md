# EDGAR offline fixtures

Real, trimmed SEC EDGAR responses for the pinned PR 4 corpus: five life-sciences filers'
FY2019-FY2021 10-Ks (15 filings), served by `EdgarClient(..., offline=True)` so CI, the tests
and `fde rag seed-edgar --offline` never touch sec.gov.

Captured: 2026-10-23 by `python -m fde_kit.rag.adapters.capture`.

| Tenant | Company | Filings (accession, FY) |
| --- | --- | --- |
| `pharma-a` | PFIZER INC (PFE, CIK 0000078003) | 0000078003-20-000014 FY2019, 0000078003-21-000038 FY2020, 0000078003-22-000027 FY2021 |
| `pharma-a` | GILEAD SCIENCES, INC. (GILD, CIK 0000882095) | 0000882095-20-000006 FY2019, 0000882095-21-000008 FY2020, 0000882095-22-000007 FY2021 |
| `pharma-a` | VERTEX PHARMACEUTICALS INC / MA (VRTX, CIK 0000875320) | 0000875320-20-000007 FY2019, 0000875320-21-000006 FY2020, 0000875320-22-000007 FY2021 |
| `pharma-b` | Moderna, Inc. (MRNA, CIK 0001682852) | 0001682852-20-000006 FY2019, 0001682852-21-000006 FY2020, 0001682852-22-000012 FY2021 |
| `pharma-b` | REGENERON PHARMACEUTICALS, INC. (REGN, CIK 0000872589) | 0001532176-20-000008 FY2019, 0001804220-21-000008 FY2020, 0001804220-22-000007 FY2021 |

## Layout

`cache/<sha1(url)>.json` and `cache/<sha1(url)>.txt` are `EdgarClient` cache entries. Each file is
a JSON object `{"url": ..., "body": ...}`: `.json` holds a parsed JSON response, `.txt` a text
response. Document entries carry `"extracted": true` (see below).

## What is trimmed, and why

- **companyfacts** (`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, 2-5 MB each live):
  only the `FIELDS` tags (`RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`, `NetIncomeLoss`, `ResearchAndDevelopmentExpense`, `ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost`), the USD unit, `form == "10-K"`, and periods ending
  2019-2021. Every row kept is exactly what SEC returned (same keys, same values).
- **Filing documents** (the 10-K and any EX-13; Pfizer's 10-K is a shell whose financial
  statements are in EX-13): stored as extracted plain text (`html_to_text`) truncated to the
  first 40,000 characters, flagged `"extracted": true`. The sha256 of the FULL extracted text
  is pinned in `src/fde_kit/rag/adapters/edgar_manifest.json`, so the truncation is visible and
  drift in the live archive is detectable (`fde rag edgar-drift`).
- **Filing index pages** (`<accession>-index.htm`, which lists each document with its type):
  stored whole.

Facts come from the XBRL data, not from the document text, so truncation does not affect them.

## Regenerate

```
FDE_EDGAR_USER_AGENT="<app name> <contact email>" python -m fde_kit.rag.adapters.capture
```

SEC requires a contact in the User-Agent. It is read from the environment and never written to
any file. Requests are limited to 5/second and cached in `.fde/edgar-cache`, so a re-run only
re-fetches what is missing there. Regenerating rewrites this directory and the manifest; review
the diff (a changed hash means SEC's copy changed).
