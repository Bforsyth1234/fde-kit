"""The agent-facing search tool, bound to one tenant per request.

The tenant comes from the authenticated request (`tenant_id` here), never from the model. The
tool's schema has `query` and `k` and nothing else, so there is no argument a prompt injection
could fill in to search somewhere else. Extra arguments the model invents are dropped by the
input model before the function runs; tests/test_rag_isolation.py sends one anyway.

Build a fresh registry per request:

    registry = tenant_search_registry(store, tenant_id=request.tenant)
    Agent(registry).run(request.question)
"""

from __future__ import annotations

from ..tools import ToolRegistry
from .schema import validate_tenant_id
from .store import TenantStore

MAX_K = 10


def tenant_search_registry(
    store: TenantStore,
    tenant_id: str,
    registry: ToolRegistry | None = None,
    *,
    cached: bool = True,
) -> ToolRegistry:
    validate_tenant_id(tenant_id)
    if registry is not None and registry.tenant_id not in (None, tenant_id):
        raise ValueError(f"registry already belongs to tenant {registry.tenant_id!r}")
    # Never mutate a caller's shared base registry (where tenant_id is None) in place: doing so
    # would bind that shared instance and its `search_documents` tool to this request's tenant.
    if registry is None:
        out = ToolRegistry(tenant_id=tenant_id)
    elif registry.tenant_id is None:
        out = ToolRegistry(_tools=dict(registry._tools), tenant_id=tenant_id)
    else:
        out = registry

    @out.tool(name="search_documents")
    def search_documents(query: str, k: int = 5) -> str:
        """Search this customer's documents (leases, policies, addenda) and return the most
        relevant passages with their source. Use it for any question about terms, amounts,
        dates or rules; quote the source you used."""
        k = max(1, min(int(k), MAX_K))
        with store.session(tenant_id) as s:
            hits = s.cached_search(query, k=k).hits if cached else s.search(query, k=k)
        if not hits:
            return "No matching passages."
        return "\n\n---\n\n".join(f"[{h.source}]\n{h.content}" for h in hits)

    return out
