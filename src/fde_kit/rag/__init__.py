"""Multi-tenant retrieval: Postgres + pgvector, isolation enforced by Row-Level Security.

Optional extra: `pip install 'fde-kit[rag]'`. See README "Multi-tenant retrieval".
"""

from .chunk import chunk_text
from .embed import Embedder, FastEmbedEmbedder, HashEmbedder, make_embedder
from .schema import (
    APP_ROLE,
    InsecureConnection,
    PostureProblem,
    app_url_from_admin,
    check_posture,
    create_tenant,
    drop_tenant,
    migrate,
)
from .sqlguard import SqlRejected, check_select
from .store import (
    DictEntry,
    DocSlice,
    Fact,
    FactRow,
    Hit,
    IngestOutcome,
    QueryResult,
    SearchResult,
    TenantSession,
    TenantStore,
    values_agree,
)
from .tool import tenant_search_registry

__all__ = [
    "APP_ROLE",
    "DictEntry",
    "DocSlice",
    "Embedder",
    "Fact",
    "FactRow",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "Hit",
    "IngestOutcome",
    "InsecureConnection",
    "PostureProblem",
    "QueryResult",
    "SearchResult",
    "SqlRejected",
    "TenantSession",
    "TenantStore",
    "app_url_from_admin",
    "check_posture",
    "check_select",
    "chunk_text",
    "create_tenant",
    "drop_tenant",
    "make_embedder",
    "migrate",
    "tenant_search_registry",
    "values_agree",
]
