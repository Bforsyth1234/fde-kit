"""SEC EDGAR: pinned 10-K filings as documents, their XBRL financials as facts.

Network access goes through `EdgarClient`: the contact SEC asks for in the User-Agent, a rate
limit well under SEC's fair-access ceiling, and an on-disk cache that doubles as the offline
fixture store (`evals/fixtures/edgar/cache`).

Cache entries are `<sha1(url)>.json` (a JSON response) or `<sha1(url)>.txt` (a text
response); both files hold a JSON object `{"url": ..., "body": ...}` where `body` is the parsed
JSON or the text. A document entry written by the capture script also has `"extracted": true`:
its body is already plain text (truncated), not HTML. The User-Agent is never stored.
"""

from __future__ import annotations

import gzip
import hashlib
import json as _json
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..store import DictEntry
from ._factrow import make_fact_row
from .base import AdapterBatch, RawDocument, RawEntity, RawFact

SEC_WWW = "https://www.sec.gov"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
MAX_REQUESTS_PER_SECOND = 5
FIRST_YEAR, LAST_YEAR = 2019, 2021
"""Periods ending in these fiscal years are kept."""

FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "rd_expense": (
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ),
}

FIELD_DEFINITIONS = {
    "revenue": "Total revenue for the fiscal year, as reported in the 10-K's XBRL financials.",
    "net_income": "Net income (loss) attributable to the company for the fiscal year.",
    "rd_expense": "Research and development expense for the fiscal year.",
}


class EdgarCacheMiss(LookupError):
    """An offline client was asked for a URL that is not in its cache."""


@dataclass(frozen=True)
class Filing:
    cik: str
    ticker: str
    company: str
    accession: str
    form: str
    fiscal_year: int
    period_end: date
    filed: date

    @property
    def archive_dir(self) -> str:
        return f"{SEC_WWW}/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}"

    @property
    def index_url(self) -> str:
        """The filing's index page: one row per document with its type (10-K, EX-13, ...)."""
        return f"{self.archive_dir}/{self.accession}-index.htm"


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def write_cache_entry(
    cache_dir: Path, url: str, body: Any, *, kind: str, extracted: bool = False
) -> Path:
    """Write one cache entry. kind is "json" or "txt"."""
    if kind not in ("json", "txt"):
        raise ValueError(f"cache kind must be 'json' or 'txt', got {kind!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {"url": url}
    if extracted:
        entry["extracted"] = True
    entry["body"] = body
    path = cache_dir / f"{cache_key(url)}.{kind}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json.dumps(entry, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


class EdgarClient:
    """HTTP to sec.gov with the contact SEC asks for, a rate limit and an on-disk cache.

    - user_agent: required, non-empty (settings.edgar_user_agent / FDE_EDGAR_USER_AGENT);
      construction raises ValueError when empty. Never logged, never written to any file.
    - max 5 requests/second (SEC allows 10; stay well under)
    - cache_dir: GET responses cached by URL; `offline=True` serves only from cache and raises
      `EdgarCacheMiss` on a miss (and needs no user_agent)
    - stdlib urllib only (honours HTTPS_PROXY); no new dependency
    """

    def __init__(self, user_agent: str, *, cache_dir: Path, offline: bool = False):
        if not offline and not (user_agent or "").strip():
            raise ValueError(
                "EdgarClient needs a contact User-Agent for sec.gov (SEC fair-access policy): "
                "set FDE_EDGAR_USER_AGENT, e.g. 'my-app you@example.com', or use offline=True "
                "with a fixture cache"
            )
        self._user_agent = (user_agent or "").strip()
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.requests = 0
        """Network requests made (cache hits excluded)."""

    def __repr__(self) -> str:  # never show the contact
        return f"EdgarClient(cache_dir={str(self.cache_dir)!r}, offline={self.offline})"

    # ------------------------------------------------------------------ public

    def json(self, url: str, *, refresh: bool = False) -> Any:
        return self._get(url, "json", refresh=refresh)["body"]

    def text(self, url: str, *, refresh: bool = False) -> str:
        return self._get(url, "txt", refresh=refresh)["body"]

    def document_text(self, url: str, *, refresh: bool = False) -> str:
        """Plain text of a filing document: the cached extraction when the entry was written
        already extracted (offline fixtures), else `html_to_text` of the fetched HTML."""
        entry = self._get(url, "txt", refresh=refresh)
        return entry["body"] if entry.get("extracted") else html_to_text(entry["body"])

    def cached(self, url: str, kind: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{cache_key(url)}.{kind}"
        if not path.exists():
            return None
        entry = _json.loads(path.read_text(encoding="utf-8"))
        if entry.get("url") != url:  # sha1 collision or a hand-edited file
            return None
        return entry

    # ------------------------------------------------------------------ internals

    def _get(self, url: str, kind: str, *, refresh: bool) -> dict[str, Any]:
        if not refresh:
            entry = self.cached(url, kind)
            if entry is not None:
                return entry
        if self.offline:
            raise EdgarCacheMiss(
                f"offline EDGAR client: {url} is not in the cache at {self.cache_dir} "
                "(regenerate fixtures with `python -m fde_kit.rag.adapters.capture`, or go online)"
            )
        raw = self._fetch(url)
        body: Any = _json.loads(raw) if kind == "json" else raw
        write_cache_entry(self.cache_dir, url, body, kind=kind)
        return {"url": url, "body": body}

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + 1.0 / MAX_REQUESTS_PER_SECOND - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _fetch(self, url: str) -> str:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        for attempt in range(4):
            self._throttle()
            self.requests += 1
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                    encoding = (resp.headers.get("Content-Encoding") or "").lower()
                    charset = resp.headers.get_content_charset() or "utf-8"
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"GET {url} failed: HTTP {e.code}") from None
            if encoding == "gzip":
                data = gzip.decompress(data)
            elif encoding == "deflate":
                data = zlib.decompress(data)
            return data.decode(charset, errors="replace")
        raise RuntimeError(f"GET {url} failed after retries")  # pragma: no cover


# ---------------------------------------------------------------------- HTML -> text

_SKIP = {"script", "style", "head", "ix:header"}
_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "center", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody",
    "thead", "tfoot", "title", "ul", "caption",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buf: list[str] = []
        self.skip = 0
        self.row: list[list[str]] | None = None
        """Cells of the open <tr>, each a list of text pieces."""

    def _flush(self) -> None:
        line = " ".join("".join(self.buf).split())
        if line:
            self.lines.append(line)
        self.buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "tr":
            self._flush()
            self.row = []
        elif tag in ("td", "th"):
            if self.row is None:  # a cell outside any row: treat as a block
                self._flush()
            else:
                self.row.append([])
        elif tag in _BLOCK:
            self._block_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.skip and tag in _BLOCK:
            self._block_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "tr" and self.row is not None:
            cells = [" ".join("".join(c).split()) for c in self.row]
            line = " | ".join(c for c in cells if c)
            if line:
                self.lines.append(line)
            self.row = None
        elif tag in _BLOCK and tag != "br":
            self._block_break()

    def _block_break(self) -> None:
        if self.row is not None:
            if self.row:
                self.row[-1].append(" ")  # a block inside a cell is just a space
        else:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.row is not None:
            if not self.row:
                self.row.append([])
            self.row[-1].append(data)
        else:
            self.buf.append(data)

    def text(self) -> str:
        if self.row is not None:
            self.handle_endtag("tr")
        self._flush()
        return "\n".join(self.lines)


def html_to_text(html: str) -> str:
    """Plain text from (inline XBRL) HTML: script/style/head/ix:header dropped, block tags as
    line breaks, a table row's non-empty cells joined by " | ", whitespace (including nbsp)
    collapsed, blank lines dropped. Stdlib html.parser; deterministic."""
    p = _TextExtractor()
    p.feed(html)
    p.close()
    return p.text()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- filings


def company_facts(client: EdgarClient, cik: str) -> dict:
    """The XBRL companyfacts JSON for a CIK (10-digit, zero-padded or not)."""
    return client.json(COMPANYFACTS_URL.format(cik=str(int(cik)).zfill(10)))


class _IndexParser(HTMLParser):
    """Rows of the "Document Format Files" table on a filing's -index.htm page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.done = False
        self.rows: list[tuple[list[str], str | None]] = []
        self._cells: list[str] | None = None
        self._href: str | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table" and not self.done and a.get("summary") == "Document Format Files":
            self.in_table = True
        elif not self.in_table:
            return
        elif tag == "tr":
            self._cells, self._href = [], None
        elif tag in ("td", "th") and self._cells is not None:
            self._cell = []
        elif tag == "a" and self._cell is not None and self._href is None:
            self._href = a.get("href")

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag == "table":
            self.in_table, self.done = False, True
        elif tag in ("td", "th") and self._cells is not None and self._cell is not None:
            self._cells.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._cells is not None:
            self.rows.append((self._cells, self._href))
            self._cells = None

    def handle_data(self, data: str) -> None:
        if self.in_table and self._cell is not None:
            self._cell.append(data)


def parse_filing_index(html: str) -> list[tuple[str, str]]:
    """(absolute document url, document type) for every document in a filing index page."""
    p = _IndexParser()
    p.feed(html)
    p.close()
    out: list[tuple[str, str]] = []
    for cells, href in p.rows:
        # Seq | Description | Document | Type | Size
        if href is None or len(cells) < 4:
            continue
        path = href.split("?doc=", 1)[1] if "?doc=" in href else href
        url = path if path.startswith("http") else f"{SEC_WWW}{path}"
        out.append((url, cells[3]))
    return out


def filing_documents(client: EdgarClient, filing: Filing, *, refresh: bool = False):
    """(url, doc_type) for the primary document (type == filing.form) and any EX-13, in that
    order, from the filing's index page in the accession's archive directory. Never from
    submissions "recent" (older filings of frequent filers fall out of that window)."""
    docs = parse_filing_index(client.text(filing.index_url, refresh=refresh))
    primary = [(u, t) for u, t in docs if t == filing.form]
    if len(primary) != 1:
        raise ValueError(
            f"{filing.accession}: expected one {filing.form} document in {filing.index_url}, "
            f"found {len(primary)}"
        )
    ex13 = [(u, t) for u, t in docs if t.upper().startswith("EX-13")]
    return primary + ex13


def infer_precision(value: int) -> int:
    """Largest power of ten in {1, 10, ..., 10**6} that divides value; 0 -> 1. Values are whole
    dollars in companyfacts; a later filing's $0.1M/$1M rounding shows as 10**5/10**6."""
    v = abs(int(value))
    if v == 0:
        return 1
    p = 1
    while p < 10**6 and v % (p * 10) == 0:
        p *= 10
    return p


def _d(s: str) -> date:
    return date.fromisoformat(s)


def xbrl_facts(cf: dict, filing: Filing, fields: dict[str, tuple[str, ...]]) -> list[tuple]:
    """(field_name, FactRow) for full-year periods (350–372 days) ending in FY2019–FY2021 that
    THIS filing (accn) reports; the first tag in the priority tuple that the filing has for that
    period; unit USD; precision = infer_precision(val); source_ref = f"us-gaap:{tag}@{accn}".
    Rows are ordered by field (as in `fields`), then period."""
    gaap = cf.get("facts", {}).get("us-gaap", {})
    entity = cf["entityName"]
    out: list[tuple] = []
    for name, tags in fields.items():
        chosen: dict[tuple[str, str], tuple[str, int]] = {}
        for tag in tags:
            for r in gaap.get(tag, {}).get("units", {}).get("USD", []):
                if r.get("accn") != filing.accession or "start" not in r:
                    continue
                start, end = _d(r["start"]), _d(r["end"])
                if not 350 <= (end - start).days <= 372:
                    continue
                if not FIRST_YEAR <= end.year <= LAST_YEAR:
                    continue
                key = (r["start"], r["end"])
                if key in chosen and chosen[key][0] != tag:
                    continue  # an earlier-priority tag already covers this period
                chosen.setdefault(key, (tag, int(r["val"])))
        for (start, end), (tag, val) in sorted(chosen.items()):
            out.append(
                (
                    name,
                    make_fact_row(
                        entity=entity,
                        field=name,
                        value=val,
                        unit="USD",
                        period_start=_d(start),
                        period_end=_d(end),
                        precision=infer_precision(val),
                        source_ref=f"us-gaap:{tag}@{filing.accession}",
                    ),
                )
            )
    return out


def dictionary_entries(fields: dict[str, tuple[str, ...]] = FIELDS) -> list[DictEntry]:
    return [
        DictEntry(
            field=name,
            definition=FIELD_DEFINITIONS.get(name, name),
            unit="USD",
            notes=(
                "From SEC XBRL companyfacts; per filing, the first of these us-gaap tags the "
                "filing reports: " + ", ".join(tags) + ". precision = the rounding the filing "
                "used (a later filing may repeat a prior year rounded coarser)."
            ),
        )
        for name, tags in fields.items()
    ]


class EdgarAdapter:
    """One batch per company: entity (kind "company", canonical = companyfacts entityName,
    aliases = (ticker, CIK without leading zeros)); one RawDocument per filing document
    (source = archive URL); facts cited to the EX-13 when the filing has one, else the primary
    document; dictionary entries for the FIELDS. max_doc_chars truncates text; None = full."""

    name = "edgar"

    def __init__(
        self, client: EdgarClient, filings: list[Filing], *, max_doc_chars: int | None = None
    ):
        self.client = client
        self.filings = list(filings)
        self.max_doc_chars = max_doc_chars

    def batches(self) -> Iterator[AdapterBatch]:
        by_cik: dict[str, list[Filing]] = {}
        for f in self.filings:
            by_cik.setdefault(str(int(f.cik)), []).append(f)
        for cik, filings in by_cik.items():
            cf = company_facts(self.client, cik)
            company = cf["entityName"]
            entity = RawEntity("company", company, tuple(dict.fromkeys((filings[0].ticker, cik))))
            documents: list[RawDocument] = []
            facts: list[RawFact] = []
            for filing in sorted(filings, key=lambda f: (f.fiscal_year, f.accession)):
                docs = filing_documents(self.client, filing)
                for url, doc_type in docs:
                    text = self.client.document_text(url)
                    if self.max_doc_chars is not None:
                        text = text[: self.max_doc_chars]
                    documents.append(
                        RawDocument(
                            source=url,
                            title=f"{company} {filing.form} FY{filing.fiscal_year} ({doc_type})",
                            text=text,
                            metadata={
                                "cik": cik,
                                "ticker": filing.ticker,
                                "accession": filing.accession,
                                "form": filing.form,
                                "fiscal_year": str(filing.fiscal_year),
                                "doc_type": doc_type,
                                "filed": filing.filed.isoformat(),
                            },
                        )
                    )
                cite = next((u for u, t in docs if t.upper().startswith("EX-13")), docs[0][0])
                facts.extend(RawFact(cite, row) for _, row in xbrl_facts(cf, filing, FIELDS))
            yield AdapterBatch(
                entities=[entity],
                documents=documents,
                facts=facts,
                dictionary=dictionary_entries(),
            )
