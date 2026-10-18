"""One real round-trip through the provider boundary.

Everything else in the suite uses the mock, which cannot catch SDK signature changes, auth
wiring, or request-shape errors. This test makes a single cheap call and is skipped unless an
API key is available, so it runs in CI on `main` and nowhere it can't.
"""

import os

import pytest

from fde_kit.agent import Agent
from fde_kit.builtin_tools import registry
from fde_kit.config import Settings

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


def test_live_tool_round_trip():
    settings = Settings(provider="anthropic", otel_enabled=False, max_tokens=256)
    result = Agent(registry, settings=settings).run("Use the calculator to compute 17 * 23.")
    assert "calculator" in result.tools_called
    assert "391" in result.output
    assert result.usage.input_tokens > 0
