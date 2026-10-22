"""Embedders. Two of them, for two different jobs.

- `HashEmbedder`: feature hashing over word unigrams and bigrams. Deterministic, no download, no
  network, ~instant. Texts that share words land near each other, which is all the isolation
  tests need (a near-duplicate in another tenant must be *close*, or the test proves nothing).
  It is a test and offline-eval tool, not a retrieval model.
- `FastEmbedEmbedder`: BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU). 384 dimensions, ~130 MB
  on first use, cached after. This is the one the retrieval numbers in the README come from.

Both produce L2-normalised vectors of `EMBED_DIM` so the schema doesn't care which is in use.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .schema import EMBED_DIM

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class HashEmbedder:
    name = "hash"
    dim = EMBED_DIM

    def _embed(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        feats = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        v = [0.0] * self.dim
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        return _normalise(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbedder:
    dim = EMBED_DIM

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised by hand
            raise RuntimeError("fastembed not installed: pip install 'fde-kit[rag]'") from exc
        self.name = model
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def make_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name in ("bge", "fastembed"):
        return FastEmbedEmbedder()
    return FastEmbedEmbedder(name)


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
