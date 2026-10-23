"""Capture the pinned EDGAR corpus: the manifest and the offline fixtures.

    FDE_EDGAR_USER_AGENT="my-app you@example.com" python -m fde_kit.rag.adapters.capture

Discovers each company's FY2019-FY2021 10-Ks from the `accn` values in its XBRL companyfacts
(never from submissions "recent"), fetches each filing's index page and its 10-K + EX-13
documents live, and writes:

- `src/fde_kit/rag/adapters/edgar_manifest.json`: 15 filings, each document's sha256 over the
  FULL extracted text;
- `evals/fixtures/edgar/cache/`: EdgarClient cache entries the offline client serves —
  companyfacts trimmed to the FIELDS tags / USD / 10-K / periods ending 2019-2021 (rows kept
  exactly as SEC returned them), index pages whole, documents as extracted text truncated to
  40,000 chars;
- `evals/fixtures/edgar/README.md`.

The contact is read from the environment and never written to any file. Full responses are
also kept in the live cache (settings.edgar_cache_dir), so a re-run makes no new requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from ...config import get_settings
from .edgar import (
    COMPANYFACTS_URL,
    FIELDS,
    FIRST_YEAR,
    LAST_YEAR,
    EdgarClient,
    Filing,
    company_facts,
    filing_documents,
    html_to_text,
    sha256_text,
    write_cache_entry,
)
from .manifest import EDGAR_MANIFEST

FIXTURE_CHARS = 40_000
FISCAL_YEARS = (2019, 2020, 2021)

COMPANIES: list[tuple[str, str, str]] = [
    # (tenant, ticker, CIK)
    ("pharma-a", "PFE", "0000078003"),
    ("pharma-a", "GILD", "0000882095"),
    ("pharma-a", "VRTX", "0000875320"),
    ("pharma-b", "MRNA", "0001682852"),
    ("pharma-b", "REGN", "0000872589"),
]


def discover_filings(cf: dict, ticker: str, cik: str) -> list[Filing]:
    """The company's 10-Ks for FISCAL_YEARS, by the accession numbers its XBRL facts cite."""
    seen: dict[str, dict] = {}
    for tags in FIELDS.values():
        for tag in tags:
            rows = cf["facts"].get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
            for r in rows:
                if r.get("form") != "10-K" or r.get("fy") not in FISCAL_YEARS:
                    continue
                s = seen.setdefault(r["accn"], {"fy": set(), "filed": set(), "end": set()})
                s["fy"].add(r["fy"])
                s["filed"].add(r["filed"])
                if r.get("fp") == "FY":
                    s["end"].add(r["end"])
    filings: list[Filing] = []
    for accn, s in seen.items():
        if len(s["fy"]) != 1 or len(s["filed"]) != 1:
            raise ValueError(f"{ticker} {accn}: ambiguous fy/filed {s}")
        filings.append(
            Filing(
                cik=cik,
                ticker=ticker,
                company=cf["entityName"],
                accession=accn,
                form="10-K",
                fiscal_year=next(iter(s["fy"])),
                period_end=date.fromisoformat(max(s["end"])),
                filed=date.fromisoformat(next(iter(s["filed"]))),
            )
        )
    filings.sort(key=lambda f: f.fiscal_year)
    if [f.fiscal_year for f in filings] != list(FISCAL_YEARS):
        raise ValueError(f"{ticker}: expected one 10-K per FY {FISCAL_YEARS}, got {filings}")
    return filings


def trim_company_facts(cf: dict) -> dict:
    """Only the FIELDS tags, USD, form 10-K, periods ending FIRST_YEAR-LAST_YEAR. Kept rows and
    tag metadata are exactly as SEC returned them."""
    gaap = cf["facts"].get("us-gaap", {})
    kept: dict[str, dict] = {}
    for tags in FIELDS.values():
        for tag in tags:
            if tag not in gaap or tag in kept:
                continue
            rows = [
                r
                for r in gaap[tag].get("units", {}).get("USD", [])
                if r.get("form") == "10-K" and FIRST_YEAR <= int(r["end"][:4]) <= LAST_YEAR
            ]
            if rows:
                meta = {k: v for k, v in gaap[tag].items() if k != "units"}
                kept[tag] = {**meta, "units": {"USD": rows}}
    return {"cik": cf["cik"], "entityName": cf["entityName"], "facts": {"us-gaap": kept}}


README = """# EDGAR offline fixtures

Real, trimmed SEC EDGAR responses for the pinned PR 4 corpus: five life-sciences filers'
FY2019-FY2021 10-Ks (15 filings), served by `EdgarClient(..., offline=True)` so CI, the tests
and `fde rag seed-edgar --offline` never touch sec.gov.

Captured: {captured} by `python -m fde_kit.rag.adapters.capture`.

| Tenant | Company | Filings (accession, FY) |
| --- | --- | --- |
{table}

## Layout

`cache/<sha1(url)>.json` and `cache/<sha1(url)>.txt` are `EdgarClient` cache entries. Each file is
a JSON object `{{"url": ..., "body": ...}}`: `.json` holds a parsed JSON response, `.txt` a text
response. Document entries carry `"extracted": true` (see below).

## What is trimmed, and why

- **companyfacts** (`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, 2-5 MB each live):
  only the `FIELDS` tags ({tags}), the USD unit, `form == "10-K"`, and periods ending
  {first}-{last}. Every row kept is exactly what SEC returned (same keys, same values).
- **Filing documents** (the 10-K and any EX-13; Pfizer's 10-K is a shell whose financial
  statements are in EX-13): stored as extracted plain text (`html_to_text`) truncated to the
  first {chars:,} characters, flagged `"extracted": true`. The sha256 of the FULL extracted text
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
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", type=Path, default=Path("evals/fixtures/edgar"))
    ap.add_argument("--manifest", type=Path, default=EDGAR_MANIFEST)
    ap.add_argument("--cache-dir", type=Path, default=None, help="live cache (default: settings)")
    args = ap.parse_args(argv)

    settings = get_settings()
    ua = os.environ.get("FDE_EDGAR_USER_AGENT", "") or settings.edgar_user_agent
    if not ua.strip():
        print("FDE_EDGAR_USER_AGENT is not set (SEC requires a contact).", file=sys.stderr)
        return 2
    client = EdgarClient(ua, cache_dir=args.cache_dir or Path(settings.edgar_cache_dir))
    out = args.fixtures / "cache"
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.json")) + list(out.glob("*.txt")):
        old.unlink()

    entries: list[dict] = []
    rows_for_readme: list[str] = []
    for tenant, ticker, cik in COMPANIES:
        cf = company_facts(client, cik)
        url = COMPANYFACTS_URL.format(cik=cik)
        write_cache_entry(out, url, trim_company_facts(cf), kind="json")
        filings = discover_filings(cf, ticker, cik)
        for f in filings:
            write_cache_entry(out, f.index_url, client.text(f.index_url), kind="txt")
            documents: dict[str, str] = {}
            for doc_url, doc_type in filing_documents(client, f):
                full = html_to_text(client.text(doc_url))
                documents[doc_url] = sha256_text(full)
                write_cache_entry(out, doc_url, full[:FIXTURE_CHARS], kind="txt", extracted=True)
                print(f"{ticker} FY{f.fiscal_year} {doc_type:6} {len(full):>9,} chars  {doc_url}")
            entries.append(
                {
                    "tenant": tenant,
                    "ticker": ticker,
                    "cik": cik,
                    "company": f.company,
                    "accession": f.accession,
                    "form": f.form,
                    "fiscal_year": f.fiscal_year,
                    "period_end": f.period_end.isoformat(),
                    "filed": f.filed.isoformat(),
                    "documents": documents,
                }
            )
        rows_for_readme.append(
            f"| `{tenant}` | {cf['entityName']} ({ticker}, CIK {cik}) | "
            + ", ".join(f"{f.accession} FY{f.fiscal_year}" for f in filings)
            + " |"
        )

    captured = date.today().isoformat()
    manifest = {
        "description": (
            "Pinned EDGAR corpus for fde-kit PR 4. documents: url -> sha256 of the full "
            "extracted text (html_to_text). Written by python -m fde_kit.rag.adapters.capture."
        ),
        "captured": captured,
        "fixture_chars": FIXTURE_CHARS,
        "filings": entries,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tags = ", ".join(f"`{t}`" for tags in FIELDS.values() for t in tags)
    (args.fixtures / "README.md").write_text(
        README.format(
            captured=captured,
            table="\n".join(rows_for_readme),
            tags=tags,
            first=FIRST_YEAR,
            last=LAST_YEAR,
            chars=FIXTURE_CHARS,
        ),
        encoding="utf-8",
    )
    print(f"{len(entries)} filings; {client.requests} network requests; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
