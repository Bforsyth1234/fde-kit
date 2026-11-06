"""`fde_kit.tools.load_registry`: how the CLI takes a customer's tools (`--tools`)."""

import json
import sys
from pathlib import Path

import pytest

from fde_kit import ToolRegistry
from fde_kit.cli import main as cli_main
from fde_kit.tools import load_registry

TOOLS_SRC = '''
from fde_kit import ToolRegistry

reg = ToolRegistry()


@reg.tool
def freight_quote(weight_lb: float, zone: str) -> str:
    """Freight in USD."""
    return f"${35 + weight_lb * {'A': 0.18, 'B': 0.27}[zone]:.2f}"


def make():
    return reg


not_a_registry = 42
'''


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    (tmp_path / "acme_tools_mod.py").write_text(TOOLS_SRC)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield tmp_path
    sys.modules.pop("acme_tools_mod", None)


def test_module_attr_resolves_from_cwd(project):
    reg = load_registry("acme_tools_mod:reg")
    assert isinstance(reg, ToolRegistry) and reg.names() == ["freight_quote"]
    assert reg.execute("freight_quote", {"weight_lb": 200, "zone": "B"}).content == "$89.00"


def test_file_path_and_factory(project):
    assert load_registry(f"{project / 'acme_tools_mod.py'}:make").names() == ["freight_quote"]


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("acme_tools_mod", "expected module:attr"),
        ("acme_tools_mod:nope", "has no attribute 'nope'"),
        ("acme_tools_mod:not_a_registry", "expected a ToolRegistry"),
        ("missing_dir/x.py:reg", "no such file"),
    ],
)
def test_bad_specs_say_why(project, spec, match):
    with pytest.raises(ValueError, match=match):
        load_registry(spec)


def test_fde_tools_lists_custom_tools(project, capsys):
    assert cli_main(["tools", "--tools", "acme_tools_mod:reg"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["name"] == "freight_quote"


def test_fde_tools_bad_spec_exits_2(project, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["tools", "--tools", "acme_tools_mod:nope"])
    assert exc.value.code == 2
    assert "has no attribute" in capsys.readouterr().out


def test_fde_run_uses_custom_tools(project, monkeypatch):
    """`fde run --tools` hands the agent the customer's registry instead of the builtin one."""
    import fde_kit.cli as cli

    seen = {}

    class StopAgent(Exception):
        pass

    class FakeAgent:
        def __init__(self, registry, **_):
            seen["names"] = registry.names()

        def run(self, *_a, **_k):
            raise StopAgent

    monkeypatch.setenv("FDE_OTEL_ENABLED", "false")
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    with pytest.raises(StopAgent):
        cli_main(["run", "--tools", "acme_tools_mod:reg", "hi"])
    assert seen["names"] == ["freight_quote"]
