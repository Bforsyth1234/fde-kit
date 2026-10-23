"""Source adapters: external sources become documents and facts with provenance (PR 4)."""

from .base import (
    Adapter,
    AdapterBatch,
    IngestReport,
    RawDocument,
    RawEntity,
    RawFact,
    ingest,
)
from .edgar import (
    FIELDS,
    EdgarAdapter,
    EdgarCacheMiss,
    EdgarClient,
    Filing,
    company_facts,
    filing_documents,
    html_to_text,
    infer_precision,
    xbrl_facts,
)
from .local_files import LocalFilesAdapter
from .manifest import EDGAR_MANIFEST, ManifestEntry, check_drift, filings_for, load_manifest

__all__ = [
    "EDGAR_MANIFEST",
    "FIELDS",
    "Adapter",
    "AdapterBatch",
    "EdgarAdapter",
    "EdgarCacheMiss",
    "EdgarClient",
    "Filing",
    "IngestReport",
    "LocalFilesAdapter",
    "ManifestEntry",
    "RawDocument",
    "RawEntity",
    "RawFact",
    "check_drift",
    "company_facts",
    "filing_documents",
    "filings_for",
    "html_to_text",
    "infer_precision",
    "ingest",
    "load_manifest",
    "xbrl_facts",
]
