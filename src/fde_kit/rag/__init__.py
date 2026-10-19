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
    migrate,
)
from .store import Hit, SearchResult, TenantSession, TenantStore
from .tool import tenant_search_registry

__all__ = [
    "APP_ROLE",
    "Embedder",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "Hit",
    "InsecureConnection",
    "PostureProblem",
    "SearchResult",
    "TenantSession",
    "TenantStore",
    "app_url_from_admin",
    "check_posture",
    "chunk_text",
    "create_tenant",
    "make_embedder",
    "migrate",
    "tenant_search_registry",
]
