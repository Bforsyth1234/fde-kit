"""The semantic cache everyone writes first, kept here so the test can show why it's wrong.

Process-wide, keyed on the query embedding, no notion of who asked. It's fast, it's simple, and
two customers who ask the same question get the same answer: the first one's.
"""

from __future__ import annotations

from fde_kit.rag.embed import Embedder
from fde_kit.rag.store import Hit, TenantSession


class NaiveSemanticCache:
    def __init__(self, embedder: Embedder, max_distance: float = 0.08):
        self.embedder = embedder
        self.max_distance = max_distance
        self._entries: list[tuple[list[float], list[Hit]]] = []

    def search(self, session: TenantSession, query: str, k: int = 5) -> tuple[list[Hit], bool]:
        q = self.embedder.embed_query(query)
        for emb, hits in self._entries:
            dist = 1 - sum(a * b for a, b in zip(q, emb, strict=True))
            if dist < self.max_distance:
                return hits, True
        hits = session.search(query, k=k)
        self._entries.append((q, hits))
        return hits, False
