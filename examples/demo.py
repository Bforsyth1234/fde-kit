"""Smallest end-to-end example: register a custom tool, run the agent, print the trace summary.

    export ANTHROPIC_API_KEY=...
    docker compose up -d            # optional: Phoenix at http://localhost:6006
    python examples/demo.py
"""

from fde_kit import Agent, ToolRegistry, init_tracing
from fde_kit.builtin_tools import registry as builtin

init_tracing()

registry = ToolRegistry()


@registry.tool
def shipping_estimate(weight_kg: float, express: bool = False) -> str:
    """Estimate shipping cost in USD for a parcel. Express doubles the rate."""
    base = 4.5 + 1.2 * weight_kg
    return f"${base * (2 if express else 1):.2f}"


# Reuse the built-in calculator alongside the custom tool.
registry.tool(builtin.get("calculator").fn)

agent = Agent(registry)
result = agent.run(
    "How much is express shipping for a 3.5 kg parcel, and what would three of them cost?"
)

print(result.output)
print(f"\nturns={result.turns} tools={result.tools_called} stopped={result.stopped_reason}")
print(f"tokens in/out = {result.usage.input_tokens}/{result.usage.output_tokens}")
