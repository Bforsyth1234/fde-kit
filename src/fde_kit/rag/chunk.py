"""Chunking. Deliberately simple: split on blank lines, keep headings with the text under them,
pack paragraphs up to `max_chars`, and prefix every chunk with the document title.

The title prefix matters more than any chunking cleverness for documents that share a template.
Two leases from the same property manager differ in a handful of values; without the unit and
tenant name on every chunk, "the late fee clause" is the same text in every lease and retrieval
can only guess which one you meant.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6}\s|\d+(\.\d+)*[.)]\s|[A-Z][A-Z0-9 &/,-]{3,}$)")


def chunk_text(text: str, title: str = "", max_chars: int = 900) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            body = "\n\n".join(buf)
            chunks.append(f"{title}\n\n{body}" if title else body)
        buf, size = [], 0

    for p in paras:
        is_heading = bool(_HEADING.match(p.splitlines()[0]))
        if buf and (is_heading or size + len(p) > max_chars):
            flush()
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            buf.append(p[:cut].strip())
            flush()
            p = p[cut:].strip()
        buf.append(p)
        size += len(p)
    flush()
    return chunks
