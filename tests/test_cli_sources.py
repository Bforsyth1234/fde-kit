"""`fde run` / `fde rag ask` sources output: full source locations, copyable at any width."""

import io

import pytest
from rich.console import Console

from fde_kit import cli
from fde_kit.tools import Source

URL = (
    "https://www.sec.gov/Archives/edgar/data/78003/000007800320000014/"
    "pfe-exhibit13x12312019.htm"
)


class Result:
    def __init__(self, sources, cited=(), fabricated=()):
        self.sources = list(sources)
        self.cited = list(cited)
        self.fabricated_citations = list(fabricated)


def _render(result, width: int) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=width, force_terminal=False, color_system=None)
    original = cli.console
    cli.console = console
    try:
        cli._print_sources(result)
    finally:
        cli.console = original
    return buf.getvalue()


@pytest.mark.parametrize("width", [60, 80, 120])
def test_full_url_is_printed_on_one_line_at_any_width(width):
    src = Source(source_id="f1", uri=URL, title="PFIZER INC revenue 2019", content_hash="a" * 64)
    out = _render(Result([src], cited=[src]), width)
    assert any(line.strip() == f"1  {URL}" for line in out.splitlines()), out
    assert "…" not in out.split("\n" + "1  ")[-1]  # the link line itself is never truncated


def test_table_keeps_number_cited_title_and_hash():
    a = Source(
        source_id="f1", uri=URL, title="PFIZER INC revenue 2019", content_hash="0d1f7f6a8379ff"
    )
    b = Source(source_id="c9", uri="harbor/leases/unit-4b.md", title="")
    out = _render(Result([a, b], cited=[a]), 120)
    assert "PFIZER INC revenue 2019" in out and "0d1f7f6a8379" in out and "yes" in out
    assert "harbor/leases/unit-4b.md" in out  # a title-less source shows its location instead
    lines = out.splitlines()
    assert lines[-2].strip() == f"1  {URL}" and lines[-1].strip() == "2  harbor/leases/unit-4b.md"


def test_numbers_are_right_aligned_past_nine_sources():
    srcs = [
        Source(source_id=f"s{i}", uri=f"https://example.com/{i}", title=f"t{i}") for i in range(12)
    ]
    out = _render(Result(srcs), 120)
    assert " 1  https://example.com/0" in out and "12  https://example.com/11" in out


def test_nothing_printed_without_sources():
    assert _render(Result([]), 80) == ""
