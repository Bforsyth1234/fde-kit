"""The pinned EDGAR corpus: 5 companies x FY2019-FY2021 10-Ks, in two tenants.

`edgar_manifest.json` is written by `python -m fde_kit.rag.adapters.capture`. Each entry pins a
filing by accession and records the sha256 of the FULL extracted text of each of its documents,
so the offline fixtures (truncated) and later drift in the live archive are both detectable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .edgar import EdgarClient, Filing, filing_documents, sha256_text

EDGAR_MANIFEST = Path(__file__).with_name("edgar_manifest.json")


@dataclass(frozen=True)
class ManifestEntry:
    tenant: str
    ticker: str
    cik: str
    accession: str
    form: str
    fiscal_year: int
    documents: dict[str, str] = field(default_factory=dict)
    """Document url -> sha256 of the FULL extracted text."""
    company: str = ""
    period_end: str = ""
    filed: str = ""

    def filing(self) -> Filing:
        return Filing(
            cik=self.cik,
            ticker=self.ticker,
            company=self.company,
            accession=self.accession,
            form=self.form,
            fiscal_year=self.fiscal_year,
            period_end=date.fromisoformat(self.period_end),
            filed=date.fromisoformat(self.filed),
        )


def load_manifest(path: Path = EDGAR_MANIFEST) -> list[ManifestEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ManifestEntry(
            tenant=e["tenant"],
            ticker=e["ticker"],
            cik=e["cik"],
            accession=e["accession"],
            form=e["form"],
            fiscal_year=int(e["fiscal_year"]),
            documents=dict(e["documents"]),
            company=e.get("company", ""),
            period_end=e["period_end"],
            filed=e["filed"],
        )
        for e in data["filings"]
    ]


def tenants(path: Path = EDGAR_MANIFEST) -> list[str]:
    return list(dict.fromkeys(e.tenant for e in load_manifest(path)))


def filings_for(tenant: str, path: Path = EDGAR_MANIFEST) -> list[Filing]:
    """The pinned filings of one tenant, in manifest order."""
    return [e.filing() for e in load_manifest(path) if e.tenant == tenant]


def check_drift(client: EdgarClient, path: Path = EDGAR_MANIFEST) -> list[str]:
    """Re-fetch every pinned filing's index and documents (bypassing the cache), re-extract,
    and compare with the manifest. Returns human-readable messages; empty = no drift. Needs an
    online client; the fresh responses are written to its cache, so point it at the live cache
    (settings.edgar_cache_dir), never at the committed fixtures."""
    if client.offline:
        raise ValueError("check_drift needs an online EdgarClient (it re-fetches from sec.gov)")
    problems: list[str] = []
    for e in load_manifest(path):
        label = f"{e.ticker} FY{e.fiscal_year} {e.form} {e.accession}"
        try:
            docs = filing_documents(client, e.filing(), refresh=True)
        except Exception as exc:  # noqa: BLE001 - report, keep checking the rest
            problems.append(f"{label}: index could not be read: {exc}")
            continue
        urls = [u for u, _ in docs]
        for url in sorted(set(e.documents) - set(urls)):
            problems.append(f"{label}: pinned document no longer listed: {url}")
        for url in sorted(set(urls) - set(e.documents)):
            problems.append(f"{label}: new document listed: {url}")
        for url in urls:
            if url not in e.documents:
                continue
            digest = sha256_text(client.document_text(url, refresh=True))
            if digest != e.documents[url]:
                problems.append(
                    f"{label}: extracted text changed: {url} "
                    f"(pinned {e.documents[url][:12]}, now {digest[:12]})"
                )
    return problems
